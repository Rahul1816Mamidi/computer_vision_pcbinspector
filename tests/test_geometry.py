import numpy as np

from src.utils.geometry import iou_matrix, match_image


def test_iou_identical_and_disjoint():
    a = [[0, 0, 10, 10]]
    assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0.0


def test_match_tp_fp_fn():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    m = match_image(gt, [0, 1], [[1, 1, 10, 10], [200, 200, 210, 210]], [0, 2], [0.9, 0.8])
    assert m["tp"] == [(0, 0)]
    assert m["fp"] == [1]
    assert m["fn"] == [1]
    assert m["confused"] == []


def test_match_confusion_and_no_stealing():
    gt = [[0, 0, 10, 10]]
    # wrong-class prediction with HIGHER confidence must not steal the GT from the correct-class one
    m = match_image(gt, [0], [[0, 0, 10, 10], [0, 0, 10, 10]], [3, 0], [0.99, 0.6])
    assert m["tp"] == [(1, 0)]
    assert m["fp"] == [0]
    # only a wrong-class prediction -> confusion
    m2 = match_image(gt, [0], [[0, 0, 10, 10]], [3], [0.9])
    assert m2["confused"] == [(0, 0)] and m2["tp"] == [] and m2["fn"] == []


def test_empty_inputs():
    m = match_image(np.zeros((0, 4)), [], [[0, 0, 5, 5]], [0], [0.5])
    assert m["fp"] == [0] and m["fn"] == []
