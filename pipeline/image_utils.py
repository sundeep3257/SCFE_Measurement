"""Image loading, validation, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from PIL import Image, UnidentifiedImageError

from pipeline.config import MIN_IMAGE_DIM
from pipeline.coordinates import CropBox, LetterboxMeta


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def load_png_array(path: Path) -> np.ndarray:
    """
    Load a PNG into a 2D float32 grayscale array without EXIF-based rotation.

    Supports grayscale, RGB, RGBA, and 16-bit PNG when Pillow can decode them.
    """
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode in ("I;16", "I"):
                arr = np.asarray(img, dtype=np.float32)
            else:
                arr = np.asarray(img.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not read image file: {path.name}") from exc

    if arr.ndim != 2:
        raise ImageValidationError("Expected a 2D grayscale image after conversion.")

    if arr.size == 0:
        raise ImageValidationError("Image is empty.")

    h, w = arr.shape
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise ImageValidationError(
            f"Image dimensions ({w}×{h}) are below the minimum ({MIN_IMAGE_DIM} px)."
        )

    return arr


def validate_png_upload(path: Path) -> None:
    """Validate extension and decodable PNG contents."""
    if path.suffix.lower() != ".png":
        raise ImageValidationError("Only PNG files are supported.")
    load_png_array(path)


def png_to_nifti(png_path: Path, nifti_path: Path | None = None) -> Path:
    """
    Convert an uploaded PNG to a 2D NIfTI volume using the project's standard
    orientation convention (transpose to X×Y storage, singleton Z, RAS affine).

    This matches the batch conversion script used to prepare training data.
    """
    png_path = Path(png_path)
    if nifti_path is None:
        nifti_path = png_path.with_suffix(".nii")
    else:
        nifti_path = Path(nifti_path)

    try:
        img = Image.open(png_path)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not read image file: {png_path.name}") from exc

    if img.mode == "RGBA":
        img = img.convert("RGB")
    if img.mode not in ("L", "I", "I;16"):
        img = img.convert("L")

    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float32)
    else:
        arr = np.array(img, dtype=np.float32)

    if arr.ndim != 2:
        raise ImageValidationError("Expected a 2D image after grayscale conversion.")

    h, w = arr.shape
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise ImageValidationError(
            f"Image dimensions ({w}×{h}) are below the minimum ({MIN_IMAGE_DIM} px)."
        )

    # PNG (row, col) → NIfTI (X, Y) storage used by the trained models.
    arr = arr.T
    arr = arr[:, :, np.newaxis]

    affine = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    nii = nib.Nifti1Image(arr, affine)
    nii.header.set_zooms((1.0, 1.0, 1.0))
    nii.header.set_xyzt_units("mm")
    nib.save(nii, str(nifti_path))
    return nifti_path


def load_nifti_raw_2d(path: Path) -> np.ndarray:
    """
    Load a 2D NIfTI slice without transposing (Region 1 / segmentation convention).

    Returns array with shape (X, Y) as stored in the file.
    """
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ImageValidationError(f"Expected 2D NIfTI, got shape {arr.shape}.")
    return arr


def load_nifti_display_2d(path: Path) -> np.ndarray:
    """
    Load a 2D NIfTI slice and transpose to display/landmark convention
    (Regions 2, 3, 5 — matches Implement_Pipeline.py load_nii_as_array / load_image_as_array).
    """
    return load_nifti_raw_2d(path).T


def to_display_space(arr_nifti: np.ndarray) -> np.ndarray:
    """Map an array from NIfTI storage (X, Y) to display space (H, W)."""
    return arr_nifti.T


def to_nifti_space(arr_display: np.ndarray) -> np.ndarray:
    """Map an array from display space (H, W) to NIfTI storage (X, Y)."""
    return arr_display.T


def normalize_percentile(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Robust percentile normalization used by the segmentation model."""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [low, high])
    if p_high > p_low:
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low)
    else:
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
    return img.astype(np.float32)


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    arr = arr.astype(np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def compute_roi_crop_box(
    arr: np.ndarray,
    threshold: float = 0.1,
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
) -> CropBox:
    """Bounding box around bright ROI with optional padding."""
    arr_norm = normalize_minmax(arr)
    mask = arr_norm > threshold
    h, w = arr_norm.shape
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return CropBox(0, 0, w - 1, h - 1)

    roi_x0, roi_x1 = int(np.min(xs)), int(np.max(xs))
    roi_y0, roi_y1 = int(np.min(ys)), int(np.max(ys))

    return CropBox(
        x0=max(0, roi_x0 - pad_left),
        y0=max(0, roi_y0 - pad_top),
        x1=min(w - 1, roi_x1 + pad_right),
        y1=min(h - 1, roi_y1 + pad_bottom),
    )


def crop_array(arr: np.ndarray, box: CropBox) -> np.ndarray:
    return arr[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]


def letterbox_resize(arr: np.ndarray, target_size: int) -> Tuple[np.ndarray, LetterboxMeta]:
    """Aspect-preserving resize with zero padding to a square."""
    h, w = arr.shape
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    resized = np.asarray(pil).astype(np.float32) / 255.0

    padded = np.zeros((target_size, target_size), dtype=np.float32)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    meta = LetterboxMeta(scale=scale, pad_left=pad_left, pad_top=pad_top, target_size=target_size)
    return padded, meta


def array_to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float grayscale array to 8-bit for annotation backgrounds."""
    norm = normalize_minmax(arr)
    return (norm * 255).astype(np.uint8)


def make_display_combined_mask(label_mask: np.ndarray) -> np.ndarray:
    """Convert class labels (0/1/2) to display values (0/128/255)."""
    from pipeline.config import HEAD_DISPLAY_VALUE, SHAFT_DISPLAY_VALUE

    display = np.zeros_like(label_mask, dtype=np.uint8)
    display[label_mask == 1] = HEAD_DISPLAY_VALUE
    display[label_mask == 2] = SHAFT_DISPLAY_VALUE
    return display
