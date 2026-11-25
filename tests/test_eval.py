import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from click.testing import CliRunner
from pyproj import CRS
from rasterio.transform import Affine, from_origin

import ftw_tools.training.eval as evaluation
from ftw_tools.cli import model_test
from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.eval import expand_countries
from ftw_tools.training.metrics import (
    FP_DEPENDENT_SCORES,
    chamfer_statistics,
    instance_statistics,
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


def _write_reference(path, mask, *, crs="EPSG:32632", transform=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        height=mask.shape[0],
        width=mask.shape[1],
        dtype=mask.dtype,
        crs=crs,
        transform=transform if transform is not None else from_origin(100, 500, 10, 8),
    ) as target:
        target.write(mask, 1)


@pytest.mark.parametrize("crs", ["EPSG:32632", "EPSG:2263", "EPSG:4326"])
def test_reference_pixel_spacing_in_metres(tmp_path, crs):
    path = tmp_path / "mask.tif"
    mask = np.zeros((4, 4), dtype=np.uint8)
    transform = (
        from_origin(30, 50, 0.0001, 0.0002)
        if crs == "EPSG:4326"
        else from_origin(100, 500, 10, 8)
    )
    _write_reference(path, mask, crs=crs, transform=transform)
    actual = evaluation._reference_pixel_spacing(path, mask.shape)
    projection = CRS.from_user_input(crs)
    if projection.is_geographic:
        x, y = transform * (2, 2)
        geod = projection.get_geod()
        expected = (
            geod.inv(x, y, x, y + transform.e)[2],
            geod.inv(x, y, x + transform.a, y)[2],
        )
    else:
        factor = projection.axis_info[0].unit_conversion_factor
        expected = (8 * factor, 10 * factor)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("crs", "transform", "shape"),
    [
        (None, from_origin(100, 500, 10, 8), (4, 4)),
        ("EPSG:32632", Affine(10, 1, 100, 0, -8, 500), (4, 4)),
        ("EPSG:32632", from_origin(100, 500, 10, 8), (3, 4)),
        ("EPSG:4326", from_origin(30, 100, 0.01, 0.01), (4, 4)),
    ],
)
def test_reference_pixel_spacing_rejects_unsupported_grids(
    tmp_path, crs, transform, shape
):
    path = tmp_path / "mask.tif"
    _write_reference(
        path, np.zeros((4, 4), dtype=np.uint8), crs=crs, transform=transform
    )
    with pytest.raises(ValueError):
        evaluation._reference_pixel_spacing(path, shape)


def _mock_checkpoint_test(monkeypatch, tmp_path, *, size=4, ignored_prediction=False):
    masks = np.zeros((3, size, size), dtype=np.uint8)
    masks[:, 1:3, 1:3] = 1
    predictions = masks.copy()
    predictions[1] = np.roll(predictions[1], 1, axis=1)
    predictions[2] = 0
    if ignored_prediction:
        masks[0, 0, 0] = 3
        predictions[0, 0, 0] = 1
    filenames = []
    for i, mask in enumerate(masks):
        path = tmp_path / f"reference_{i}.tif"
        _write_reference(
            path, mask, transform=from_origin(100, 500, 10 * (i + 1), 8 * (i + 1))
        )
        filenames.append({"mask": str(path)})

    class Dataset:
        def __init__(self, **kwargs):
            self.filenames = filenames

        def __len__(self):
            return len(self.filenames)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = 0

        def forward(self, images):
            labels = torch.from_numpy(
                predictions[self.offset : self.offset + len(images)].astype(np.int64)
            )
            self.offset += len(images)
            logits = torch.nn.functional.one_hot(labels, 2).permute(0, 3, 1, 2).float()
            return torch.nn.functional.interpolate(
                logits.to(images.device), size=images.shape[-2:], mode="nearest"
            )

    trainer = SimpleNamespace(
        model=Model(), hparams={"model": "unet", "ignore_index": 3}
    )
    monkeypatch.setattr(evaluation, "FTW", Dataset)
    monkeypatch.setattr(
        evaluation,
        "CustomSemanticSegmentationTask",
        SimpleNamespace(load_from_checkpoint=lambda *args, **kwargs: trainer),
    )
    monkeypatch.setattr(
        evaluation,
        "DataLoader",
        lambda ds, **kwargs: [
            {
                "image": torch.zeros((len(batch), 8, size, size)),
                "mask": torch.from_numpy(batch.astype(np.int64)),
            }
            for batch in (masks[:2], masks[2:])
        ],
    )
    return masks, predictions, filenames


def _run_checkpoint_test(tmp_path, **kwargs):
    arguments = {
        "model_path": "model.ckpt",
        "dir": str(tmp_path),
        "gpu": -1,
        "countries": ["germany"],
        "iou_threshold": 0.5,
        "out": str(tmp_path / "metrics.csv"),
        "model_predicts_3_classes": False,
        "test_on_3_classes": False,
        "temporal_options": "stacked",
        "use_val_set": False,
        "swap_order": False,
        "norm_constant": None,
        "resize_factor": 1,
        "num_workers": 0,
    }
    arguments.update(kwargs)
    evaluation.test(**arguments)
    if arguments["out"] is None:
        return None
    with open(arguments["out"], newline="") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize("bootstrap", [False, True])
def test_default_evaluation_does_not_run_extended_engine(
    monkeypatch, tmp_path, capsys, bootstrap
):
    _mock_checkpoint_test(monkeypatch, tmp_path)
    real_bootstrap = evaluation.bootstrap_confidence_intervals
    monkeypatch.setattr(
        evaluation,
        "bootstrap_confidence_intervals",
        lambda *args, **kwargs: real_bootstrap(*args, **kwargs, n_bootstrap=3),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Opt-in extended path ran during default evaluation")

    for name in (
        "instance_statistics",
        "chamfer_statistics",
        "_reference_pixel_spacing",
    ):
        monkeypatch.setattr(evaluation, name, unexpected)
    rows = _run_checkpoint_test(tmp_path, bootstrap=bootstrap)
    assert len(rows) == 1
    assert not any(name.startswith("extended_") for name in rows[0])
    assert float(rows[0]["object_level_recall"]) == pytest.approx(1 / 3)
    assert "extended_sq:" not in capsys.readouterr().out


def test_extended_evaluation_can_report_without_csv(monkeypatch, tmp_path, capsys):
    _mock_checkpoint_test(monkeypatch, tmp_path)
    _run_checkpoint_test(tmp_path, extended_object_metrics=True, out=None)
    assert "extended_best_iou_chamfer_m:" in capsys.readouterr().out
    assert not (tmp_path / "metrics.csv").exists()


def test_extended_evaluation_appends_matching_csv_schema(monkeypatch, tmp_path):
    for _ in range(2):
        _mock_checkpoint_test(monkeypatch, tmp_path)
        rows = _run_checkpoint_test(tmp_path, extended_object_metrics=True)
    assert len(rows) == 2
    assert rows[0] == rows[1]


def test_extended_evaluation_masks_predictions_on_ignored_pixels(monkeypatch, tmp_path):
    _mock_checkpoint_test(monkeypatch, tmp_path, ignored_prediction=True)
    row = _run_checkpoint_test(tmp_path, extended_object_metrics=True)[0]
    assert float(row["object_level_precision"]) == pytest.approx(1 / 3)
    assert float(row["extended_object_precision"]) == pytest.approx(1 / 2)


def test_mixed_country_pool_keeps_presence_only_restrictions(monkeypatch, tmp_path):
    _mock_checkpoint_test(monkeypatch, tmp_path)
    row = _run_checkpoint_test(
        tmp_path, countries=["germany", "rwanda"], extended_object_metrics=True
    )[0]
    assert math.isnan(float(row["extended_pq"]))
    assert float(row["extended_sq"]) == 1


@pytest.mark.parametrize("country", ["germany", "rwanda"])
def test_extended_evaluation_pools_and_reports_scores(
    monkeypatch, tmp_path, capsys, country
):
    masks, predictions, filenames = _mock_checkpoint_test(monkeypatch, tmp_path)
    rows = _run_checkpoint_test(
        tmp_path, countries=[country], extended_object_metrics=True
    )
    stats = [instance_statistics(mask, pred) for mask, pred in zip(masks, predictions)]
    distances = [
        chamfer_statistics(
            mask,
            pred,
            pixel_size_m=evaluation._reference_pixel_spacing(
                record["mask"], mask.shape
            ),
        )
        for mask, pred, record in zip(masks, predictions, filenames)
    ]
    expected = evaluation._extended_object_scores(
        stats, distances, background_known=country == "germany"
    )
    assert len(rows) == 1
    for name, value in expected.items():
        assert float(rows[0][name]) == pytest.approx(value, nan_ok=True)
    assert float(rows[0]["extended_recovery"]) == pytest.approx(4 / 9)
    assert float(rows[0]["extended_threshold_chamfer_m_count"]) == 1
    assert float(rows[0]["extended_best_iou_chamfer_m_count"]) == 2
    # Legacy scores remain defined even when the expanded suite suppresses them.
    assert math.isfinite(float(rows[0]["object_level_precision"]))
    if country == "rwanda":
        for name in FP_DEPENDENT_SCORES:
            assert math.isnan(float(rows[0]["extended_" + name]))
    assert "extended_sq:" in capsys.readouterr().out


def test_extended_resize_preserves_final_single_sample_batch(monkeypatch, tmp_path):
    _mock_checkpoint_test(monkeypatch, tmp_path, size=256)
    rows = _run_checkpoint_test(tmp_path, resize_factor=2, extended_object_metrics=True)
    assert float(rows[0]["extended_reference_count"]) == 3
    assert float(rows[0]["extended_object_recall"]) == pytest.approx(1 / 3)


def test_extended_scores_pool_sums_not_chip_means():
    distances = [
        {
            "threshold_chamfer_m_sum": 2,
            "threshold_chamfer_m_count": 1,
            "best_iou_chamfer_m_sum": 2,
            "best_iou_chamfer_m_count": 1,
            "reference_count": 1,
            "best_iou_reference_count": 1,
        },
        {
            "threshold_chamfer_m_sum": 24,
            "threshold_chamfer_m_count": 3,
            "best_iou_chamfer_m_sum": 24,
            "best_iou_chamfer_m_count": 3,
            "reference_count": 5,
            "best_iou_reference_count": 3,
        },
    ]
    scores = evaluation._extended_object_scores([], distances, background_known=True)
    assert scores["extended_threshold_chamfer_m"] == pytest.approx(26 / 4)
    assert scores["extended_best_iou_reference_coverage"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("background_known", [False, True])
def test_extended_bootstrap_resamples_chip_statistics(monkeypatch, background_known):
    masks = [np.ones((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    predictions = [masks[0].copy(), np.ones((2, 2), dtype=np.uint8)]
    stats = [instance_statistics(mask, pred) for mask, pred in zip(masks, predictions)]
    distances = [
        chamfer_statistics(mask, pred, pixel_size_m=10)
        for mask, pred in zip(masks, predictions)
    ]
    draws = iter((np.array([0, 0]), np.array([1, 1])))
    monkeypatch.setattr(
        evaluation.np.random, "choice", lambda *args, **kwargs: next(draws)
    )
    point, intervals = evaluation.bootstrap_confidence_intervals(
        predictions,
        masks,
        [1, 0],
        [0, 1],
        [0, 0],
        n_bootstrap=2,
        instance_stats=stats,
        boundary_stats=distances,
        background_known=background_known,
    )
    if background_known:
        assert point["extended_object_precision"] == 0.5
        assert intervals["extended_object_precision"] == pytest.approx((0.025, 0.975))
    else:
        assert math.isnan(point["extended_object_precision"])
        assert all(
            math.isnan(value) for value in intervals["extended_object_precision"]
        )
    assert intervals["extended_threshold_chamfer_m"] == (0, 0)
    assert point["extended_reference_count"] == 1


@pytest.mark.parametrize(
    ("stats", "distances"), [([], None), (None, []), ([], []), ([None], [None, None])]
)
def test_bootstrap_rejects_misaligned_extended_statistics(stats, distances):
    masks = [np.zeros((2, 2), dtype=np.uint8)]
    with pytest.raises(ValueError, match="both|align"):
        evaluation.bootstrap_confidence_intervals(
            masks, masks, [0], [0], [0], instance_stats=stats, boundary_stats=distances
        )


def test_extended_evaluation_writes_bootstrap_intervals(monkeypatch, tmp_path):
    _mock_checkpoint_test(monkeypatch, tmp_path)
    real_bootstrap = evaluation.bootstrap_confidence_intervals

    def short_bootstrap(*args, **kwargs):
        return real_bootstrap(*args, **kwargs, n_bootstrap=3)

    monkeypatch.setattr(evaluation, "bootstrap_confidence_intervals", short_bootstrap)
    rows = _run_checkpoint_test(tmp_path, extended_object_metrics=True, bootstrap=True)
    assert len(rows) == 1
    assert float(rows[0]["extended_sq_ci_lower"]) <= float(
        rows[0]["extended_sq_ci_upper"]
    )
    assert float(rows[0]["extended_object_recall"]) == pytest.approx(1 / 3)
    assert "object_level_f1_ci_upper" in rows[0]
    assert all(value is not None and value != "" for value in rows[0].values())


@pytest.mark.parametrize("extended", [False, True])
def test_evaluation_rejects_mixed_csv_schemas_before_loading_model(tmp_path, extended):
    path = tmp_path / "metrics.csv"
    other_fields = evaluation._test_csv_fields(
        extended_object_metrics=not extended, bootstrap=False
    )
    before = ",".join(other_fields) + "\n"
    path.write_text(before)
    with pytest.raises(ValueError, match="separate output file"):
        _run_checkpoint_test(tmp_path, extended_object_metrics=extended)
    assert path.read_text() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_model_test_forwards_extended_flag(monkeypatch, tmp_path, enabled):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(
        evaluation, "test", lambda *args, **kwargs: calls.append(kwargs)
    )
    arguments = [
        "--model",
        str(checkpoint),
        "--countries",
        "germany",
        "--dir",
        str(tmp_path),
    ]
    if enabled:
        arguments.append("--extended_object_metrics")
    result = CliRunner().invoke(model_test, arguments)
    assert result.exit_code == 0, result.output
    assert calls == [{"extended_object_metrics": enabled}]
