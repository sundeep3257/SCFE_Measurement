"""Tests for landmark mask splitting logic."""

import numpy as np

from pipeline.landmarks import _split_head_shaft_masks


def test_split_normalized_display_mask_finds_shaft():
    """Normalized 0/128/255 must not be mistaken for class labels 0/1/2."""
    display = np.zeros((100, 100), dtype=np.uint8)
    display[10:40, 10:40] = 128
    display[50:90, 20:80] = 255

    seg_norm = (display.astype(np.float32) - display.min()) / (display.max() - display.min())
    head, shaft = _split_head_shaft_masks(seg_norm)

    assert head.sum() > 0
    assert shaft.sum() > 0


def test_split_class_label_mask():
    labels = np.zeros((50, 50), dtype=np.uint8)
    labels[5:20, 5:20] = 1
    labels[25:45, 10:40] = 2

    head, shaft = _split_head_shaft_masks(labels.astype(np.float32))

    assert head.sum() > 0
    assert shaft.sum() > 0
