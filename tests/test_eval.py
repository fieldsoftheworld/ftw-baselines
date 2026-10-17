import tracemalloc

import numpy as np
import pytest
import torch
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.eval import (
    bootstrap_confidence_intervals,
    expand_countries,
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


def _legacy_bootstrap(outputs, masks, object_counts, n_bootstrap, confidence_level):
    np.random.seed(42)
    n_samples = len(outputs)
    resampled = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Precision(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Recall(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
            ]
        )
        metrics.update(outputs[indices], masks[indices])
        results = metrics.compute()
        tps, fps, fns = object_counts[indices].sum(axis=0)
        precision = tps / (tps + fps)
        recall = tps / (tps + fns)
        resampled.append(
            {
                "pixel_iou": results["MulticlassJaccardIndex"][1].item(),
                "pixel_precision": results["MulticlassPrecision"][1].item(),
                "pixel_recall": results["MulticlassRecall"][1].item(),
                "object_precision": precision,
                "object_recall": recall,
                "object_f1": 2 * precision * recall / (precision + recall),
            }
        )

    alpha = 1 - confidence_level
    return {
        name: tuple(
            np.percentile(
                [m[name] for m in resampled], [100 * alpha / 2, 100 * (1 - alpha / 2)]
            )
        )
        for name in resampled[0]
    }


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n_samples, size = 24, 16
    outputs = rng.integers(0, 2, (n_samples, size, size))
    masks = rng.choice([0, 1, 3], (n_samples, size, size))
    density = rng.random(n_samples)[:, None, None]
    masks[rng.random((n_samples, size, size)) < density] = 1
    object_counts = rng.integers(1, 6, (n_samples, 3))
    return torch.from_numpy(outputs), torch.from_numpy(masks), object_counts


def test_bootstrap_matches_legacy_implementation(toy_dataset):
    outputs, masks, object_counts = toy_dataset
    expected = _legacy_bootstrap(
        outputs, masks, object_counts, n_bootstrap=50, confidence_level=0.9
    )

    sample_counts = np.hstack([pixel_confusion_counts(outputs, masks), object_counts])
    actual = bootstrap_confidence_intervals(
        sample_counts, n_resamples=50, confidence_level=0.9, seed=42
    )

    assert actual.keys() == expected.keys()
    for name, bounds in expected.items():
        assert actual[name] == pytest.approx(bounds, abs=1e-6), name


def test_bootstrap_is_reproducible_for_fixed_seed(toy_dataset):
    outputs, masks, object_counts = toy_dataset
    sample_counts = np.hstack([pixel_confusion_counts(outputs, masks), object_counts])

    first = bootstrap_confidence_intervals(sample_counts, n_resamples=100, seed=7)
    second = bootstrap_confidence_intervals(sample_counts, n_resamples=100, seed=7)
    other = bootstrap_confidence_intervals(sample_counts, n_resamples=100, seed=8)

    assert first == second
    assert first != other


def test_bootstrap_memory_is_bounded_for_large_dataset():
    rng = np.random.default_rng(0)
    sample_counts = rng.integers(0, 65_536, (200_000, 6))

    tracemalloc.start()
    try:
        intervals = bootstrap_confidence_intervals(sample_counts, n_resamples=200)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < sample_counts.nbytes
    assert all(np.isfinite(bounds).all() for bounds in intervals.values())
