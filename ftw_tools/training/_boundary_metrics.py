"""Explicit spatial boundary diagnostics; not enabled by training defaults.

Definition checked against taylor-geospatial/fields-of-the-planet,
revision a626650d2944773b48b9ddd43d914405e6d95cae, src/ftw_planet/eval.py.
This is an independent implementation, not transferred upstream code.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.spatial import cKDTree


def boundary_chamfer(first, second, *, pixel_size_m):
    """Symmetric mean unsquared distance, with scalar or (row, column) spacing.

    Matches FTP's interior EDT < 1.5 boundary rule, including its array-edge
    behavior (no exterior padding). Holes contribute boundaries. Empty boundaries
    yield NaN. Valid-domain clipping must be performed before this function.
    """
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("Masks must be aligned 2D arrays")
    spacing = np.asarray(pixel_size_m, dtype=float)
    if (
        spacing.shape not in ((), (2,))
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("pixel_size_m must be finite and positive")
    boundaries = [
        mask & (distance_transform_edt(mask) < 1.5) for mask in (first, second)
    ]
    if any(not boundary.any() for boundary in boundaries):
        return float("nan")
    a, b = boundaries
    directed = (
        distance_transform_edt(~a, sampling=pixel_size_m)[b].mean(),
        distance_transform_edt(~b, sampling=pixel_size_m)[a].mean(),
    )
    return float(np.mean(directed))


def chamfer_statistics(
    truth,
    prediction,
    *,
    pixel_size_m,
    reference_labels=None,
    iou_threshold=0.5,
    prediction_connectivity=4,
    reference_connectivity=4,
    ignore_index=None,
):
    """Two pairings and reference coverage, preserving the existing identity rules.

    Best-IoU ties select the lowest compact prediction ID. Predictions may be
    reused across references. Zero-overlap references have no distance. Returned
    sums/counts support object-weighted pooling, not averaging chip means.
    """
    from ftw_tools.training.metrics import instance_match_labels

    spacing = np.asarray(pixel_size_m, dtype=float)
    if (
        spacing.shape not in ((), (2,))
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("pixel_size_m must be finite and positive")
    gt, pred, matched_gt, matched_pred, _ = instance_match_labels(
        truth,
        prediction,
        reference_labels=reference_labels,
        iou_threshold=iou_threshold,
        prediction_connectivity=prediction_connectivity,
        reference_connectivity=reference_connectivity,
        ignore_index=ignore_index,
    )
    reference_ids, reference_areas = np.unique(gt[gt > 0], return_counts=True)
    prediction_ids, prediction_areas = np.unique(pred[pred > 0], return_counts=True)
    pa = dict(zip(prediction_ids.tolist(), prediction_areas.tolist()))
    best = []
    for rid, area in zip(reference_ids, reference_areas):
        ids, intersections = np.unique(pred[gt == rid], return_counts=True)
        candidates = [
            (int(pid), int(inter)) for pid, inter in zip(ids, intersections) if pid > 0
        ]
        if candidates:
            pid, _ = max(
                candidates,
                key=lambda item: (item[1] / (area + pa[item[0]] - item[1]), -item[0]),
            )
            best.append((int(rid), pid))
    # Compute each object's boundary once in a bounding box with a one-pixel
    # background halo where available. Keep chip-edge behavior (no padding).
    # KD-tree nearest pixel centres are equivalent to EDT, avoiding a full-chip
    # distance transform for every pairing.
    boxes = (find_objects(gt), find_objects(pred))
    boundaries = ({}, {})

    def points(side, object_id):
        if object_id not in boundaries[side]:
            labels = (gt, pred)[side]
            box = boxes[side][object_id - 1]
            row = slice(max(0, box[0].start - 1), min(labels.shape[0], box[0].stop + 1))
            col = slice(max(0, box[1].start - 1), min(labels.shape[1], box[1].stop + 1))
            mask = labels[row, col] == object_id
            boundary = mask & (distance_transform_edt(mask) < 1.5)
            coordinates = np.argwhere(boundary).astype(float)
            coordinates += (row.start, col.start)
            coordinates *= spacing
            boundaries[side][object_id] = coordinates
        return boundaries[side][object_id]

    cache = {}

    def summarize(pairs, prefix):
        values = []
        for rid, pid in pairs:
            key = (int(rid), int(pid))
            if key not in cache:
                a, b = points(0, int(rid)), points(1, int(pid))
                cache[key] = (
                    float(
                        (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean())
                        / 2
                    )
                    if len(a) and len(b)
                    else float("nan")
                )
            values.append(cache[key])
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return {
            prefix: float(finite.mean()) if finite.size else float("nan"),
            prefix + "_sum": float(finite.sum()),
            prefix + "_count": int(finite.size),
        }

    result = summarize(zip(matched_gt, matched_pred), "threshold_chamfer_m")
    result.update(summarize(best, "best_iou_chamfer_m"))
    result.update(
        reference_count=len(reference_ids),
        best_iou_reference_count=len(best),
        best_iou_reference_coverage=len(best) / len(reference_ids)
        if len(reference_ids)
        else float("nan"),
    )
    return result
