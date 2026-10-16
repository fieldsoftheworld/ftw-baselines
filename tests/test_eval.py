import tracemalloc

import numpy as np
import pytest
import torch
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.eval import (
    COUNT_COLUMNS,
    bootstrap_confidence_intervals,
    expand_countries,
    metrics_from_counts,
    pixel_confusion_counts,
)


def test_expand_countries_with_full_data():
    """Test that expand_countries() correctly expands 'full_data' to all FULL_DATA_COUNTRIES."""

    # Test basic expansion
    result = expand_countries(["full_data"])

    # Verify that result is now the full list
    assert isinstance(result, list)
    assert len(result) == len(FULL_DATA_COUNTRIES)
    assert set(result) == set(FULL_DATA_COUNTRIES)

    # Verify specific expected countries are included
    expected_countries = ["austria", "belgium", "france", "germany", "netherlands"]
    for country in expected_countries:
        assert country in result, f"Expected country '{country}' not in expanded list"

    # Verify no unexpected values remain
    assert "full_data" not in result, "'full_data' should be replaced, not kept in list"


def test_expand_countries_without_full_data():
    """Test that expand_countries() preserves specific country names when 'full_data' is absent."""
    # Test with specific countries (no full_data)
    input_countries = ["rwanda", "kenya", "belgium"]
    result = expand_countries(input_countries)

    # Should return the same list
    assert result == input_countries
    assert len(result) == 3
    assert "rwanda" in result
    assert "kenya" in result
    assert "belgium" in result


def test_expand_countries_mixed_with_full_data():
    """Test that 'full_data' replaces the entire list when mixed with specific countries."""
    # Test when full_data is mixed with other countries
    result = expand_countries(["rwanda", "full_data", "kenya"])

    # When full_data is present, it should replace the entire list
    assert set(result) == set(FULL_DATA_COUNTRIES)
    assert len(result) == len(FULL_DATA_COUNTRIES)


def test_expand_countries_does_not_modify_original():
    """Test that expand_countries() does not modify the original input list."""
    # Test immutability
    original = ["rwanda", "kenya"]
    original_copy = original.copy()
    result = expand_countries(original)

    # Original should be unchanged
    assert original == original_copy
    assert result == original
    assert result is not original  # Should be a different object


def _pixel_metrics(outputs, masks, num_classes=2):
    """Class-1 pixel metrics the way `ftw model test` computes them."""
    metrics = MetricCollection(
        [
            JaccardIndex(
                task="multiclass",
                average="none",
                num_classes=num_classes,
                ignore_index=3,
            ),
            Precision(
                task="multiclass",
                average="none",
                num_classes=num_classes,
                ignore_index=3,
            ),
            Recall(
                task="multiclass",
                average="none",
                num_classes=num_classes,
                ignore_index=3,
            ),
        ]
    )
    metrics.update(outputs, masks)
    results = metrics.compute()
    return {
        "pixel_iou": results["MulticlassJaccardIndex"][1].item(),
        "pixel_precision": results["MulticlassPrecision"][1].item(),
        "pixel_recall": results["MulticlassRecall"][1].item(),
    }


def _object_metrics(tps, fps, fns):
    precision = tps / (tps + fps) if tps + fps > 0 else float("nan")
    recall = tps / (tps + fns) if tps + fns > 0 else float("nan")
    if precision + recall > 0 and not (np.isnan(precision) or np.isnan(recall)):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {"object_precision": precision, "object_recall": recall, "object_f1": f1}


def _legacy_bootstrap(outputs, masks, object_counts, n_bootstrap, confidence_level):
    """The bootstrap implementation before #272, which resamples full images."""
    np.random.seed(42)
    n_samples = len(outputs)
    resampled = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        metrics = _pixel_metrics(outputs[indices], masks[indices])
        metrics.update(_object_metrics(*object_counts[indices].sum(axis=0)))
        resampled.append(metrics)

    alpha = 1 - confidence_level
    intervals = {}
    for name in resampled[0]:
        values = [m[name] for m in resampled if not np.isnan(m[name])]
        if values:
            intervals[name] = (
                np.percentile(values, 100 * alpha / 2),
                np.percentile(values, 100 * (1 - alpha / 2)),
            )
        else:
            intervals[name] = (float("nan"), float("nan"))
    return intervals


@pytest.fixture
def toy_dataset():
    """Small random predictions, masks with ignored pixels, and object counts."""
    rng = np.random.default_rng(0)
    n_samples, size = 24, 16
    outputs = rng.integers(0, 2, (n_samples, size, size))
    masks = rng.choice([0, 1, 3], (n_samples, size, size))
    # Different crop density per sample so resamples give different metrics
    density = rng.random(n_samples)[:, None, None]
    masks[rng.random((n_samples, size, size)) < density] = 1
    object_counts = rng.integers(0, 6, (n_samples, 3))
    return torch.from_numpy(outputs), torch.from_numpy(masks), object_counts


def _sample_counts(outputs, masks, object_counts):
    return np.hstack([pixel_confusion_counts(outputs, masks), object_counts])


def test_pixel_confusion_counts_match_torchmetrics(toy_dataset):
    outputs, masks, _ = toy_dataset
    counts = pixel_confusion_counts(outputs, masks)

    assert counts.shape == (len(outputs), 3)
    totals = np.append(counts.sum(axis=0), [1, 1, 1])
    ours = metrics_from_counts(totals)
    for name, expected in _pixel_metrics(outputs, masks).items():
        assert ours[name] == pytest.approx(expected, abs=1e-6)


def test_pixel_confusion_counts_three_classes():
    outputs = torch.tensor([[[1, 1, 0, 2]]])
    masks = torch.tensor([[[1, 2, 1, 3]]])
    np.testing.assert_array_equal(pixel_confusion_counts(outputs, masks), [[1, 1, 1]])


def test_metrics_from_counts_undefined_values():
    metrics = metrics_from_counts(np.zeros(len(COUNT_COLUMNS)))

    # Pixel metrics follow torchmetrics (0), object metrics are NaN
    assert metrics["pixel_iou"] == 0
    assert metrics["pixel_precision"] == 0
    assert metrics["pixel_recall"] == 0
    assert np.isnan(metrics["object_precision"])
    assert np.isnan(metrics["object_recall"])
    assert np.isnan(metrics["object_f1"])


def test_bootstrap_matches_legacy_implementation(toy_dataset):
    outputs, masks, object_counts = toy_dataset
    expected = _legacy_bootstrap(
        outputs, masks, object_counts, n_bootstrap=50, confidence_level=0.9
    )

    actual = bootstrap_confidence_intervals(
        _sample_counts(outputs, masks, object_counts),
        n_resamples=50,
        confidence_level=0.9,
        seed=42,
    )

    assert actual.keys() == expected.keys()
    for name, (lower, upper) in expected.items():
        assert actual[name] == pytest.approx((lower, upper), abs=1e-6), name
    assert actual["pixel_iou"][0] < actual["pixel_iou"][1]


def test_bootstrap_is_reproducible_for_fixed_seed(toy_dataset):
    counts = _sample_counts(*toy_dataset)

    first = bootstrap_confidence_intervals(counts, n_resamples=100, seed=7)
    second = bootstrap_confidence_intervals(counts, n_resamples=100, seed=7)
    other = bootstrap_confidence_intervals(counts, n_resamples=100, seed=8)

    assert first == second
    assert first != other


def test_bootstrap_does_not_touch_global_rng(toy_dataset):
    np.random.seed(123)
    expected = np.random.random()

    np.random.seed(123)
    bootstrap_confidence_intervals(_sample_counts(*toy_dataset), n_resamples=10)

    assert np.random.random() == expected


def test_bootstrap_wider_interval_for_higher_confidence(toy_dataset):
    counts = _sample_counts(*toy_dataset)

    narrow = bootstrap_confidence_intervals(counts, confidence_level=0.5)
    wide = bootstrap_confidence_intervals(counts, confidence_level=0.99)

    assert wide["pixel_iou"][0] <= narrow["pixel_iou"][0]
    assert wide["pixel_iou"][1] >= narrow["pixel_iou"][1]


def test_bootstrap_all_nan_metric():
    counts = np.zeros((5, len(COUNT_COLUMNS)), dtype=np.int64)
    counts[:, 0] = 10

    intervals = bootstrap_confidence_intervals(counts, n_resamples=10)

    assert intervals["pixel_iou"] == (1.0, 1.0)
    assert all(np.isnan(intervals["object_f1"]))


@pytest.mark.parametrize(
    ("counts", "kwargs", "match"),
    [
        (np.zeros((0, 6)), {}, "empty"),
        (np.zeros((5, 4)), {}, "shape"),
        (np.ones((5, 6)), {"n_resamples": 0}, "n_resamples"),
        (np.ones((5, 6)), {"confidence_level": 1.0}, "confidence_level"),
        (np.ones((5, 6)), {"confidence_level": 0.0}, "confidence_level"),
    ],
)
def test_bootstrap_rejects_invalid_input(counts, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bootstrap_confidence_intervals(counts, **kwargs)


def test_bootstrap_memory_is_bounded_for_large_dataset():
    """A full FTW-sized run must not hold per-pixel data or grow with resamples."""
    n_samples, n_resamples = 200_000, 200
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 65_536, (n_samples, len(COUNT_COLUMNS)))

    tracemalloc.start()
    try:
        intervals = bootstrap_confidence_intervals(counts, n_resamples=n_resamples)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The old implementation held two 256x256 uint8 arrays per sample (~26 GB here)
    # and concatenated a full copy per resample; this needs a few O(n_samples) arrays.
    assert peak < counts.nbytes
    assert all(np.isfinite(bounds).all() for bounds in intervals.values())
