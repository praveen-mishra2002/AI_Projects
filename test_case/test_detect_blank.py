import os
import sys

import numpy as np

# ensure Trial_Codes is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Trial_Codes')))
from face_detection import detect_faces


def test_detect_blank_returns_empty():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detect_faces(img)
    assert len(faces) == 0
