"""
Coordinate-system utilities for the SCFE pipeline.

Canonical space (used for all user-facing outputs):
    - Origin at upper-left of the uploaded PNG
    - x increases to the right (column index)
    - y increases downward (row index)
    - Image shape: (height, width)

Model working space (transposed, matching legacy NIfTI .T convention):
    - Used by neck-line, landmark, and head-circle stages that were trained on
      transposed NIfTI arrays.
    - Forward:  (x_t, y_t) from canonical (x_c, y_c) via transpose:
        x_t = y_c, y_t = x_c
    - Inverse:  x_c = y_t, y_c = x_t

Additional transforms (tracked per stage):
    - ROI crop: subtract (x0, y0) going into crop; add on the way out
    - Letterbox: scale and pad_left/pad_top offsets within the 256x256 model input
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class CropBox:
    """Inclusive pixel bounds: x0 <= x <= x1, y0 <= y <= y1."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


@dataclass(frozen=True)
class LetterboxMeta:
    """Letterbox resize metadata for model-space (256x256) coordinates."""

    scale: float
    pad_left: int
    pad_top: int
    target_size: int


@dataclass(frozen=True)
class ImageSpace:
    """Documents the coordinate frame for a set of points."""

    name: str
    height: int
    width: int
    is_transposed: bool = False


def transpose_array(arr: np.ndarray) -> np.ndarray:
    """Apply the legacy pipeline transpose (swap rows/cols)."""
    return arr.T


def canonical_to_transposed(point: Point) -> Point:
    x_c, y_c = point
    return float(y_c), float(x_c)


def transposed_to_canonical(point: Point) -> Point:
    x_t, y_t = point
    return float(y_t), float(x_t)


def crop_to_full(point_crop: Point, crop: CropBox) -> Point:
    x, y = point_crop
    return float(x + crop.x0), float(y + crop.y0)


def full_to_crop(point_full: Point, crop: CropBox) -> Point:
    x, y = point_full
    return float(x - crop.x0), float(y - crop.y0)


def letterbox_to_crop(point_lb: Point, meta: LetterboxMeta) -> Point:
    """Convert model letterbox coordinates to cropped-image coordinates."""
    x_lb, y_lb = point_lb
    x_crop = (x_lb - meta.pad_left) / meta.scale
    y_crop = (y_lb - meta.pad_top) / meta.scale
    return float(x_crop), float(y_crop)


def crop_to_letterbox(point_crop: Point, meta: LetterboxMeta) -> Point:
    x_crop, y_crop = point_crop
    x_lb = x_crop * meta.scale + meta.pad_left
    y_lb = y_crop * meta.scale + meta.pad_top
    return float(x_lb), float(y_lb)


def letterbox_to_full(point_lb: Point, crop: CropBox, meta: LetterboxMeta) -> Point:
    return crop_to_full(letterbox_to_crop(point_lb, meta), crop)


def full_to_letterbox(point_full: Point, crop: CropBox, meta: LetterboxMeta) -> Point:
    return crop_to_letterbox(full_to_crop(point_full, crop), meta)


def map_points_transposed_to_canonical(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 0], out[:, 1] = points[:, 1], points[:, 0]
    return out


def clip_point(point: Point, width: int, height: int) -> Point:
    x, y = point
    return (
        float(np.clip(x, 0, width - 1)),
        float(np.clip(y, 0, height - 1)),
    )
