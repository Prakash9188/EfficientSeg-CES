"""Assignment metrics implemented with NumPy/OpenCV only (no sklearn/skimage)."""
import cv2
import numpy as np


def binary_mask(mask):
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(bool)


def boundary_map(mask):
    """Object pixels with a 4-connected background neighbour; outside is background."""
    m = binary_mask(mask)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    center = p[1:-1, 1:-1]
    return center & (~p[:-2, 1:-1] | ~p[2:, 1:-1] |
                     ~p[1:-1, :-2] | ~p[1:-1, 2:])


def within_5x5(source_boundary):
    """True where a source boundary exists in the centered 5x5 neighbourhood."""
    src = source_boundary.astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.filter2D(src, cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT) > 0


def boundary_f1(pred, truth):
    p = boundary_map(pred)
    t = boundary_map(truth)
    npred, ntrue = int(p.sum()), int(t.sum())
    if npred == 0 and ntrue == 0:
        return 1.0
    if npred == 0 or ntrue == 0:
        return 0.0
    t_near = within_5x5(t)
    p_near = within_5x5(p)
    precision = float((p & t_near).sum()) / npred
    recall = float((t & p_near).sum()) / ntrue
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def confusion_iou(pred, truth):
    p = binary_mask(pred)
    t = binary_mask(truth)
    # Classes: background=0, object=1.
    ious = []
    for cls in (False, True):
        inter = np.logical_and(p == cls, t == cls).sum()
        union = np.logical_or(p == cls, t == cls).sum()
        ious.append(float(inter / union) if union else 1.0)
    return sum(ious) / 2.0


def score_set(predictions, truths):
    if len(predictions) != len(truths) or not truths:
        raise ValueError("predictions and truths must be non-empty and have equal lengths")
    # Assignment requirement: accumulate one confusion matrix over the entire set,
    # then compute IoU for background and object and average the two class IoUs.
    intersections = [0, 0]
    unions = [0, 0]
    for pred, truth in zip(predictions, truths):
        p, t = binary_mask(pred), binary_mask(truth)
        for idx, cls in enumerate((False, True)):
            intersections[idx] += int(np.logical_and(p == cls, t == cls).sum())
            unions[idx] += int(np.logical_or(p == cls, t == cls).sum())
    class_ious = [intersections[i] / unions[i] if unions[i] else 1.0 for i in range(2)]
    miou = float(sum(class_ious) / 2.0)
    # Boundary F1 is calculated per image and averaged, as specified.
    bf1 = float(np.mean([boundary_f1(p, t) for p, t in zip(predictions, truths)]))
    q = 0.70 * miou + 0.30 * bf1
    return {"mIoU": miou, "Boundary_F1": bf1, "Q": q}


def ces_score(q, cpu_ms, model_bytes):
    if cpu_ms > 50.0:
        return 0.0
    return float(q * (1.0 - 0.20 * cpu_ms / 50.0
                      - 0.10 * min(1.0, model_bytes / 4194304.0)))
