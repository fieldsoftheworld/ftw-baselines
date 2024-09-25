import rasterio.features
import shapely.geometry


def get_object_level_metrics(y_true, y_pred, iou_threshold=0.5):
    """Get object level metrics for a single mask / prediction pair.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.
        iou_threshold (float, optional): IoU threshold for matching predictions to ground truths. Defaults to 0.5.

    Returns
        tuple (int, int, int): Number of true positives, false positives, and false negatives.
    """
    if iou_threshold < 0.5:
        raise ValueError("iou_threshold must be greater than 0.5")  # If we go lower than 0.5 then it is possible for a single prediction to match with multiple ground truths and we have to do de-duplication
    y_true_shapes = []
    for geom, val in rasterio.features.shapes(y_true):
        if val == 1:
            y_true_shapes.append(shapely.geometry.shape(geom))

    y_pred_shapes = []
    for geom, val in rasterio.features.shapes(y_pred):
        if val == 1:
            y_pred_shapes.append(shapely.geometry.shape(geom))

    tps = 0
    fns = 0
    tp_is = set()  # keep track of which of the true shapes are true positives
    tp_js = set()  # keep track of which of the predicted shapes are true positives
    fn_is = set()  # keep track of which of the true shapes are false negatives
    matched_js = set()
    for i, y_true_shape in enumerate(y_true_shapes):
        matching_j = None
        for j, y_pred_shape in enumerate(y_pred_shapes):
            if y_true_shape.intersects(y_pred_shape):
                intersection = y_true_shape.intersection(y_pred_shape)
                union = y_true_shape.union(y_pred_shape)
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    matching_j = j
                    matched_js.add(j)
                    tp_js.add(j)
                    break
        if matching_j is not None:
            tp_is.add(i)
            tps += 1
        else:
            fn_is.add(i)
            fns += 1
    fps = len(y_pred_shapes) - len(matched_js)
    fp_js = set(range(len(y_pred_shapes))) - matched_js  # compute which of the predicted shapes are false positives

    # Create masks of the true positives, false positives, and false negatives
    # tp_i_mask = rasterio.features.rasterize([y_true_shapes[i] for i in tp_is], out_shape= y_true.shape)
    # tp_j_mask = rasterio.features.rasterize([y_pred_shapes[j] for j in tp_js], out_shape= y_pred.shape)
    # fp_j_mask = rasterio.features.rasterize([y_pred_shapes[j] for j in fp_js], out_shape= y_pred.shape)
    # fn_i_mask = rasterio.features.rasterize([y_true_shapes[i] for i in fn_is], out_shape= y_true.shape)

    return (tps, fps, fns)