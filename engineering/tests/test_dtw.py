import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.recognition import dtw_distance


def test_dtw_same():
    x = np.random.randn(20, 13)
    assert dtw_distance(x, x) < 1e-8


def test_dtw_different():
    x = np.random.randn(20, 13)
    y = np.random.randn(20, 13) + 5
    assert dtw_distance(x, y) > dtw_distance(x, x)
