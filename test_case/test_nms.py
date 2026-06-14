import os
import sys

import numpy as np

# ensure Trial_Codes is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Trial_Codes')))
from face_detection import _nms


def test_nms_merges_overlapping():
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 50, 50],
        [200, 200, 30, 30],
    ])
    out = _nms(boxes, overlapThresh=0.3)
    # overlapping first two should be merged -> expect 2 boxes
    assert len(out) == 2
