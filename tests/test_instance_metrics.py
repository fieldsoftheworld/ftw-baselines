"""Synthetic checks for the reference-based metrics engine, without study data."""

import math

import numpy as np
import pytest
import rasterio.features
import shapely.geometry

from ftw_tools.training.metrics import (
    FP_DEPENDENT_SCORES,
    InstanceStatistics,
    boundary_chamfer,
    chamfer_statistics,
    get_object_level_metrics,
    instance_label_matches,
    instance_label_statistics,
    instance_match_labels,
    instance_scores_for_reporting,
    instance_statistics,
)


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[1, 1, 1]], [[1, 1, 0]]),
        ([[1, 1, 1, 0, 1, 1]], [[1, 0, 1, 0, 1, 1]]),
        ([[1, 1, 0, 1, 1]], [[1, 1, 1, 1, 1]]),
        ([[0, 0]], [[0, 0]]),
    ],
)
def test_compatibility_counts_match_original_polygon_helper(truth, prediction):
    def polygons(mask):
        return [
            shapely.geometry.shape(geometry)
            for geometry, value in rasterio.features.shapes(mask.astype(np.uint8))
            if value == 1
        ]

    truth, prediction = np.array(truth), np.array(prediction)
    references, predictions = polygons(truth), polygons(prediction)
    matched = set()
    tp = 0
    for reference in references:
        for index, predicted in enumerate(predictions):
            if (
                reference.intersection(predicted).area / reference.union(predicted).area
                > 0.5
            ):
                tp += 1
                matched.add(index)
                break
    counts = get_object_level_metrics(truth, prediction)
    assert counts == (tp, len(predictions) - len(matched), len(references) - tp)
    assert all(isinstance(value, int) for value in counts)


def test_legacy_helper_keeps_unmasked_counts_and_threshold_restriction(monkeypatch):
    from ftw_tools.training import metrics

    truth = np.array([[1, 3, 0]], dtype=np.uint8)
    prediction = np.array([[0, 1, 0]], dtype=np.uint8)
    assert instance_statistics(truth, prediction).fp == 0
    monkeypatch.setattr(
        metrics,
        "instance_statistics",
        lambda *args, **kwargs: pytest.fail("New engine called"),
    )
    assert get_object_level_metrics(truth, prediction) == (0, 1, 1)
    with pytest.raises(ValueError, match="greater than 0.5"):
        get_object_level_metrics(truth, prediction, iou_threshold=0.3)


def test_connectivity_and_match_labels():
    mask = np.array([[1, 0], [0, 1]])
    assert instance_statistics(mask, mask).tp == 2
    gt, pred, gi, pi, stats = instance_match_labels(
        mask, mask, reference_connectivity=8
    )
    assert stats == instance_statistics(mask, mask, reference_connectivity=8)
    assert (stats.tp, stats.fp, stats.fn) == (0, 2, 1)
    assert gt.max() == 1 and pred.max() == 2
    assert len(gi) == len(pi) == 0
    assert stats.scores()["recovery"] == pytest.approx(0.5)


def test_strict_threshold_and_score_formulas():
    truth, prediction = np.array([[1, 1, 1]]), np.array([[1, 1, 0]])
    assert instance_statistics(truth, prediction, iou_threshold=2 / 3).tp == 0
    scores = instance_statistics(truth, prediction).scores()
    for name in (
        "object_precision",
        "object_recall",
        "object_f1",
        "rq",
        "rand_merge_score",
    ):
        assert scores[name] == 1
    for name in ("sq", "pq", "recovery", "support"):
        assert scores[name] == pytest.approx(2 / 3)
    assert scores["rand_split_score"] == pytest.approx(4 / 9)


def test_matching_prioritizes_cardinality_over_best_individual_iou():
    truth = np.ones((1, 14), dtype=np.uint8)
    references = np.array([[1] * 10 + [2] * 4])
    predictions = np.array([[1] * 6 + [2] * 4 + [1] * 4])
    _, _, gi, pi, stats = instance_label_matches(
        truth,
        predictions,
        reference_labels=references,
        prediction_identity="preserve-ids",
        iou_threshold=0.3,
    )
    assert (stats.tp, stats.fp, stats.fn) == (2, 0, 0)
    assert stats.matched_iou == pytest.approx(0.8)
    assert set(zip(gi, pi)) == {(1, 2), (2, 1)}


def test_split_merge_and_unsupported_prediction_area():
    truth = np.array([[1, 1, 1, 0, 1, 1, 1]])
    split = instance_statistics(truth, np.array([[1, 0, 1, 0, 1, 1, 1]])).scores()
    assert split["rand_split_score"] == pytest.approx(11 / 18)
    assert split["rand_merge_score"] == 1
    merged = instance_statistics(truth, np.ones_like(truth)).scores()
    assert merged["rand_split_score"] == 1
    assert merged["rand_merge_score"] == pytest.approx(0.5)
    unsupported = instance_statistics(np.array([[1, 0, 0]]), np.array([[1, 1, 1]]))
    assert unsupported.scores()["support"] == pytest.approx(1 / 3)
    assert unsupported.scores()["rand_merge_score"] == 1


def test_ignored_pixels_are_removed_from_both_masks():
    truth = np.array([[1, 3, 1, 255, 0, 2]])
    prediction = np.ones_like(truth)
    gt, pred, _, _, stats = instance_match_labels(truth, prediction, ignore_index=2)
    assert (stats.tp, stats.fp, stats.fn) == (2, 1, 0)
    assert stats.prediction_area == 3
    assert not gt[0, [1, 3, 5]].any()
    assert not pred[0, [1, 3, 5]].any()


def test_empty_missed_and_pooled_statistics():
    empty = instance_statistics(np.zeros((2, 2)), np.zeros((2, 2)))
    assert all(math.isnan(value) for value in empty.scores().values())
    missed = instance_statistics(np.array([[1]]), np.array([[0]]))
    assert math.isnan(missed.scores()["object_precision"])
    assert math.isnan(missed.scores()["sq"])
    for name in ("object_recall", "object_f1", "pq", "recovery", "rand_merge_score"):
        assert missed.scores()[name] == 0
    false_positive = instance_statistics(np.array([[0]]), np.array([[1]]))
    assert false_positive.scores()["object_precision"] == 0
    assert false_positive.scores()["support"] == 0
    pooled = InstanceStatistics()
    pooled += empty
    pooled += missed
    pooled += instance_statistics(np.ones((2, 2)), np.ones((2, 2)))
    assert pooled.scores()["object_recall"] == 0.5
    assert pooled.scores()["recovery"] == pytest.approx(4 / 5)


def test_presence_only_reporting_does_not_change_statistics():
    stats = instance_statistics(np.array([[1, 0, 1]]), np.array([[1, 0, 0]]))
    scores = instance_scores_for_reporting(stats, background_known=False)
    for name in FP_DEPENDENT_SCORES:
        assert math.isnan(scores[name])
    assert scores["object_recall"] == 0.5
    assert stats.scores()["object_precision"] == 1
    assert instance_scores_for_reporting(stats, background_known=True) == stats.scores()


@pytest.mark.parametrize(
    ("truth", "references", "expected_fn"),
    [([[1, 0, 1]], [[2**63 + 7, 0, 2**63 + 7]], 1), ([[1, 1]], [[7, 9]], 2)],
)
def test_supplied_reference_ids_preserve_disconnected_and_touching_fields(
    truth, references, expected_fn
):
    truth = np.array(truth)
    references = np.array(references, dtype=np.uint64)
    stats = instance_statistics(truth, truth, reference_labels=references)
    assert stats.tp == 0
    assert stats.fn == expected_fn


def test_prediction_identity_and_wholly_masked_reference_ids():
    truth = np.array([[1, 255, 1, 0, 255]])
    references = np.array([[7, 7, 7, 0, 99]], dtype=np.uint64)
    predictions = np.array([[2**63 + 1] * 3 + [0, 100]], dtype=np.uint64)
    split = instance_label_statistics(truth, predictions, reference_labels=references)
    gt, pred, gi, pi, preserved = instance_label_matches(
        truth,
        predictions,
        reference_labels=references,
        prediction_identity="preserve-ids",
    )
    assert (split.tp, split.fp, split.fn) == (0, 2, 1)
    assert (preserved.tp, preserved.fp, preserved.fn) == (1, 0, 0)
    assert gt.tolist() == pred.tolist() == [[1, 0, 1, 0, 0]]
    assert gi.tolist() == pi.tolist() == [1]


@pytest.mark.parametrize(
    "references",
    [np.array([[0, 0]]), np.array([[1.0, 0.0]]), np.array([[-1, 0]]), np.array([[1]])],
)
def test_invalid_reference_labels(references):
    truth = np.array([[1, 0]])
    with pytest.raises(ValueError):
        instance_statistics(truth, truth, reference_labels=references)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iou_threshold": -0.1},
        {"iou_threshold": 1.1},
        {"prediction_connectivity": 6},
        {"reference_connectivity": 6},
    ],
)
def test_invalid_semantic_options(kwargs):
    with pytest.raises(ValueError):
        instance_statistics(np.ones((2, 2)), np.ones((2, 2)), **kwargs)


def test_invalid_arrays_and_prediction_ids():
    with pytest.raises(ValueError, match="aligned 2D"):
        instance_statistics(np.ones((2, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="aligned 2D"):
        instance_statistics(np.ones(2), np.ones(2))
    for labels in (np.array([[1.0]]), np.array([[-1]])):
        with pytest.raises(ValueError, match="nonnegative integers"):
            instance_label_statistics(np.array([[1]]), labels)
    with pytest.raises(ValueError, match="identity mode"):
        instance_label_statistics(
            np.array([[1]]), np.array([[1]]), prediction_identity="unknown"
        )


def test_chamfer_identity_offsets_holes_and_units():
    first = np.zeros((30, 30), dtype=np.uint8)
    first[3:24, 4:25] = 1
    assert boundary_chamfer(first, first, pixel_size_m=10) == 0
    assert math.isnan(boundary_chamfer(first, first * 0, pixel_size_m=10))
    second = np.roll(first, 2, axis=1)
    distance = boundary_chamfer(first, second, pixel_size_m=1)
    assert distance > 0
    assert boundary_chamfer(second, first, pixel_size_m=10) == pytest.approx(
        10 * distance
    )
    holed = first.copy()
    holed[10:15, 10:15] = 0
    assert boundary_chamfer(first, holed, pixel_size_m=1) > 0


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, (1, 0), (1, 2, 3)])
def test_invalid_boundary_spacing(scale):
    mask = np.ones((3, 3))
    for function in (boundary_chamfer, chamfer_statistics):
        with pytest.raises(ValueError, match="finite and positive"):
            function(mask, mask, pixel_size_m=scale)


def test_optimized_chamfer_matches_full_grid_with_anisotropic_spacing():
    rng = np.random.default_rng(91)
    for _ in range(5):
        truth = (rng.random((24, 24)) > 0.35).astype(np.uint8)
        prediction = (rng.random((24, 24)) > 0.35).astype(np.uint8)
        gt, pred, gi, pi, _ = instance_match_labels(truth, prediction)
        result = chamfer_statistics(truth, prediction, pixel_size_m=(8, 10))
        distances = [
            boundary_chamfer(gt == g, pred == p, pixel_size_m=(8, 10))
            for g, p in zip(gi, pi)
        ]
        finite = np.asarray(distances)[np.isfinite(distances)]
        assert result["threshold_chamfer_m_count"] == len(finite)
        assert result["threshold_chamfer_m_sum"] == pytest.approx(finite.sum())
        assert result["threshold_chamfer_m"] == pytest.approx(
            finite.mean() if len(finite) else np.nan, nan_ok=True
        )


def test_best_iou_chamfer_reuses_predictions_and_preserves_reference_ids():
    truth = np.zeros((24, 24), dtype=np.uint8)
    truth[:8, :8] = 1
    truth[12:20, :8] = 1
    prediction = truth.copy()
    prediction[7:13, 2:5] = 1
    result = chamfer_statistics(truth, prediction, pixel_size_m=(8, 10))
    first, second = truth.copy(), truth.copy()
    first[12:20, :8] = 0
    second[:8, :8] = 0
    distances = [
        boundary_chamfer(mask, prediction, pixel_size_m=(8, 10))
        for mask in (first, second)
    ]
    assert result["reference_count"] == result["best_iou_reference_count"] == 2
    assert result["best_iou_reference_coverage"] == 1
    assert result["best_iou_chamfer_m_count"] == 2
    assert result["best_iou_chamfer_m_sum"] == pytest.approx(sum(distances))
    assert result["best_iou_chamfer_m"] == pytest.approx(np.mean(distances))
    supplied = chamfer_statistics(
        truth, truth, pixel_size_m=10, reference_labels=truth.astype(np.int32) * 12
    )
    assert supplied["reference_count"] == 1
    assert supplied["best_iou_chamfer_m"] > 0


def test_chamfer_empty_missing_and_below_threshold_references():
    truth = np.zeros((30, 30), dtype=np.uint8)
    empty = chamfer_statistics(truth, truth, pixel_size_m=10)
    assert math.isnan(empty["best_iou_reference_coverage"])
    assert math.isnan(empty["best_iou_chamfer_m"])
    truth[3:10, 3:10] = 1
    truth[20:25, 20:25] = 1
    missed = chamfer_statistics(truth, truth * 0, pixel_size_m=10)
    assert missed["best_iou_reference_coverage"] == 0
    assert missed["best_iou_chamfer_m_count"] == 0
    prediction = np.zeros_like(truth)
    prediction[3:10, 8:15] = 1
    partial = chamfer_statistics(truth, prediction, pixel_size_m=10)
    assert math.isnan(partial["threshold_chamfer_m"])
    assert partial["best_iou_chamfer_m"] > 0
    assert partial["best_iou_reference_coverage"] == 0.5
