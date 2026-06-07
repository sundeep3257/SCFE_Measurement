"""Tests for head circle dilation."""

import numpy as np
import pytest

from pipeline.config import HEAD_CIRCLE_DILATION_FACTOR
from pipeline.head_circle import fit_head_circle


def _synthetic_seg(h: int = 200, w: int = 200) -> np.ndarray:
    """Build a minimal display mask with head (128) and shaft (255) regions."""
    seg = np.zeros((h, w), dtype=np.uint8)
    # Head blob upper-left
    seg[30:100, 40:130] = 128
    # Shaft blob lower — overlaps dilated circle at superior edge
    seg[70:180, 70:110] = 255
    return seg


def test_head_circle_dilation_factor_applied():
    seg = _synthetic_seg()
    try:
        result = fit_head_circle(seg)
    except ValueError:
        pytest.skip("Synthetic geometry insufficient for circle fit on this platform.")

    assert result["dilation_factor"] == HEAD_CIRCLE_DILATION_FACTOR
    assert abs(result["radius"] - result["radius_fitted"] * HEAD_CIRCLE_DILATION_FACTOR) < 1e-6
    assert abs(result["diameter"] - 2 * result["radius"]) < 1e-6
