"""Tests for PNG → NIfTI orientation conversion."""

import numpy as np
import pytest
from PIL import Image

from pipeline.image_utils import (
    ImageValidationError,
    load_nifti_display_2d,
    load_nifti_raw_2d,
    png_to_nifti,
    to_display_space,
)


def test_png_to_nifti_transpose_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.image_utils.MIN_IMAGE_DIM", 3)
    png_path = tmp_path / "test.png"
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)  # H=3, W=4
    Image.fromarray(arr).save(png_path)

    nii_path = png_to_nifti(png_path)
    raw = load_nifti_raw_2d(nii_path)
    display = load_nifti_display_2d(nii_path)

    assert raw.shape == (4, 3)  # (X, Y) = (W, H)
    assert display.shape == (3, 4)  # back to PNG (H, W)
    np.testing.assert_array_equal(display, arr.astype(np.float32))
    np.testing.assert_array_equal(to_display_space(raw), arr.astype(np.float32))


def test_png_to_nifti_rejects_tiny_image(tmp_path):
    png_path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(png_path)
    with pytest.raises(ImageValidationError):
        png_to_nifti(png_path)
