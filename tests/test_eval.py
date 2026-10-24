import numpy as np
import pytest
import torch
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

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


def test_bootstrap_matches_direct_computation():
    rng = np.random.default_rng(0)
    outputs = torch.from_numpy(rng.integers(0, 2, (24, 16, 16)))
    masks = torch.from_numpy(rng.choice([0, 1, 2, 3], (24, 16, 16)))
    object_counts = rng.integers(1, 6, (24, 3))
    sample_counts = np.hstack([pixel_confusion_counts(outputs, masks), object_counts])
    actual = bootstrap_confidence_intervals(sample_counts, 50, 0.9)

    pixel_metrics = {
        "pixel_iou": MulticlassJaccardIndex(3, average="none", ignore_index=3),
        "pixel_precision": MulticlassPrecision(3, average="none", ignore_index=3),
        "pixel_recall": MulticlassRecall(3, average="none", ignore_index=3),
    }
    resampled = {name: [] for name in actual}
    np_rng = np.random.RandomState(42)
    for _ in range(50):
        idx = np_rng.choice(24, size=24, replace=True)
        for name, metric in pixel_metrics.items():
            resampled[name].append(metric(outputs[idx], masks[idx])[1].item())
        tp, fp, fn = object_counts[idx].sum(0)
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        resampled["object_precision"].append(precision)
        resampled["object_recall"].append(recall)
        resampled["object_f1"].append(2 * precision * recall / (precision + recall))

    for name, values in resampled.items():
        assert actual[name] == pytest.approx(np.percentile(values, [5, 95])), name
