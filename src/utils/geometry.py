"""Box geometry and detection-vs-ground-truth matching (used for error analysis)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def iou_matrix(a: Sequence, b: Sequence) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_image(
    gt_xyxy: Sequence,
    gt_cls: Sequence[int],
    pred_xyxy: Sequence,
    pred_cls: Sequence[int],
    pred_conf: Sequence[float],
    iou_thr: float = 0.5,
) -> Dict[str, List]:
    """Match predictions to ground truth for one image.

    Two passes so a correct-class prediction is never "stolen" by a wrong-class one:
      1. same-class matching by confidence -> true positives (tp)
      2. leftover predictions vs leftover GT, any class -> confusions
    Anything left is a false positive (fp, hallucinated defect) or a
    false negative (fn, missed defect).

    Returns lists of (pred_idx, gt_idx) for tp/confused, and plain index
    lists for fp (pred idx) and fn (gt idx).
    """
    gt_cls = list(gt_cls)
    pred_cls = list(pred_cls)
    n_gt, n_pred = len(gt_cls), len(pred_cls)
    ious = iou_matrix(pred_xyxy, gt_xyxy)
    order = list(np.argsort(-np.asarray(pred_conf, dtype=float))) if n_pred else []

    gt_used = np.zeros(n_gt, dtype=bool)
    pred_used = np.zeros(n_pred, dtype=bool)
    tp: List = []
    confused: List = []

    for same_class_pass in (True, False):
        for p in order:
            if pred_used[p] or n_gt == 0:
                continue
            cand = ious[p].copy()
            cand[gt_used] = -1.0
            if same_class_pass:
                for g in range(n_gt):
                    if gt_cls[g] != pred_cls[p]:
                        cand[g] = -1.0
            g = int(cand.argmax())
            if cand[g] >= iou_thr:
                gt_used[g] = True
                pred_used[p] = True
                (tp if same_class_pass else confused).append((int(p), g))

    fp = [int(p) for p in range(n_pred) if not pred_used[p]]
    fn = [int(g) for g in range(n_gt) if not gt_used[g]]
    return {"tp": tp, "confused": confused, "fp": fp, "fn": fn}
