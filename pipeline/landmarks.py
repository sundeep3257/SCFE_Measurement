"""Region 3: five-point landmark detection."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from pipeline.config import (
    LANDMARK_HEAD_MAX,
    LANDMARK_IMG_SIZE,
    LANDMARK_NAMES,
    LANDMARK_PAD,
    LANDMARK_ROI_THRESHOLD,
    LANDMARK_SHAFT_MIN,
    LANDMARK_TO_MASK,
)
from pipeline.coordinates import CropBox, LetterboxMeta, letterbox_to_full
from pipeline.image_utils import compute_roi_crop_box, crop_array, letterbox_resize, normalize_minmax
from pipeline.model_registry import ModelRegistry


def _split_head_shaft_masks(seg_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a combined segmentation into head and shaft boolean masks.

    Accepts class labels (0 / 1 / 2) or display-encoded masks (0 / 128 / 255).
    Normalized display masks (0–1 floats) must use the threshold path, not labels.
    """
    seg_f = seg_arr.astype(np.float32)
    unique = np.unique(seg_f)

    # Integer class labels: must include class 2 (shaft), not normalized 1.0 floats.
    is_class_labels = (
        unique.size > 0
        and np.max(unique) <= 2
        and np.allclose(unique, np.round(unique))
        and 2 in unique.astype(int)
    )
    if is_class_labels:
        return (seg_f == 1), (seg_f == 2)

    # Raw display encoding or normalized 0/128/255 → 0/~0.5/1.0
    if np.max(unique) > 2:
        seg_norm = normalize_minmax(seg_f)
    else:
        seg_norm = seg_f

    pos = seg_norm > LANDMARK_ROI_THRESHOLD
    head = pos & (seg_norm <= LANDMARK_HEAD_MAX)
    shaft = seg_norm > LANDMARK_SHAFT_MIN
    return head.astype(bool), shaft.astype(bool)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & (~eroded)


def _snap_to_boundary(
    point_xy: np.ndarray,
    boundary: np.ndarray,
    fallback_mask: np.ndarray,
) -> np.ndarray:
    """Snap a point to the nearest mask boundary (matches original pipeline)."""
    h, w = boundary.shape
    x = int(np.clip(round(float(point_xy[0])), 0, w - 1))
    y = int(np.clip(round(float(point_xy[1])), 0, h - 1))
    target = boundary if np.any(boundary) else fallback_mask
    _, nearest = distance_transform_edt(~target, return_indices=True)
    return np.array([nearest[1, y, x], nearest[0, y, x]], dtype=np.float32)


def _snap_all(points: np.ndarray, seg_arr: np.ndarray) -> np.ndarray:
    head_mask, shaft_mask = _split_head_shaft_masks(seg_arr)
    head_bound = _mask_boundary(head_mask)
    shaft_bound = _mask_boundary(shaft_mask)
    snapped = []
    for i, name in enumerate(LANDMARK_NAMES):
        mask_type = LANDMARK_TO_MASK[name]
        if mask_type == "head":
            snapped.append(_snap_to_boundary(points[i], head_bound, head_mask))
        else:
            snapped.append(_snap_to_boundary(points[i], shaft_bound, shaft_mask))
    return np.array(snapped, dtype=np.float32)


def _decode_argmax(heatmaps: np.ndarray) -> np.ndarray:
    if heatmaps.ndim == 3:
        heatmaps = heatmaps[None, ...]
    b, l, h, w = heatmaps.shape
    coords = np.zeros((b, l, 2), dtype=np.float32)
    for bi in range(b):
        for li in range(l):
            flat_idx = int(np.argmax(heatmaps[bi, li]))
            y, x = divmod(flat_idx, w)
            coords[bi, li, 0], coords[bi, li, 1] = x, y
    return coords[0] if b == 1 else coords


def _to_full_display(
    points_lb: np.ndarray,
    crop: CropBox,
    meta: LetterboxMeta,
    full_shape: tuple[int, int],
) -> np.ndarray:
    points = points_lb.copy()
    for i in range(len(points)):
        points[i] = letterbox_to_full((float(points[i, 0]), float(points[i, 1])), crop, meta)
    h, w = full_shape
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    return points


@torch.no_grad()
def predict_landmarks(
    radiograph_display: np.ndarray,
    seg_display: np.ndarray,
    label_mask_display: np.ndarray | None = None,
    registry: ModelRegistry | None = None,
) -> Dict:
    """
    Detect five landmarks in display-space coordinates (H×W, matching PNG view).

    radiograph_display and seg_display must already be in the transposed display
    convention produced by load_nifti_display_2d / to_display_space.
    """
    registry = registry or ModelRegistry.get()
    model = registry.landmarks
    device = registry.device

    full_rad = normalize_minmax(radiograph_display.astype(np.float32))
    full_seg = normalize_minmax(seg_display.astype(np.float32))

    crop_box = compute_roi_crop_box(
        full_seg,
        threshold=LANDMARK_ROI_THRESHOLD,
        pad_top=LANDMARK_PAD,
        pad_bottom=LANDMARK_PAD,
        pad_left=LANDMARK_PAD,
        pad_right=LANDMARK_PAD,
    )
    crop_rad = normalize_minmax(crop_array(full_rad, crop_box))
    crop_seg = normalize_minmax(crop_array(full_seg, crop_box))

    rad_lb, meta = letterbox_resize(crop_rad, LANDMARK_IMG_SIZE)
    seg_lb, _ = letterbox_resize(crop_seg, LANDMARK_IMG_SIZE)
    tensor = torch.tensor(
        np.stack([rad_lb, seg_lb], axis=0), dtype=torch.float32
    ).unsqueeze(0).to(device)

    preds_lb = []
    pred_hm = model(tensor)[0].detach().cpu().numpy()
    preds_lb.append(_decode_argmax(pred_hm))

    tensor_h = torch.flip(tensor, dims=[3])
    pred_h = _decode_argmax(model(tensor_h)[0].detach().cpu().numpy())
    pred_h[:, 0] = LANDMARK_IMG_SIZE - 1 - pred_h[:, 0]
    preds_lb.append(pred_h)

    tensor_v = torch.flip(tensor, dims=[2])
    pred_v = _decode_argmax(model(tensor_v)[0].detach().cpu().numpy())
    pred_v[:, 1] = LANDMARK_IMG_SIZE - 1 - pred_v[:, 1]
    preds_lb.append(pred_v)

    tensor_hv = torch.flip(tensor, dims=[2, 3])
    pred_hv = _decode_argmax(model(tensor_hv)[0].detach().cpu().numpy())
    pred_hv[:, 0] = LANDMARK_IMG_SIZE - 1 - pred_hv[:, 0]
    pred_hv[:, 1] = LANDMARK_IMG_SIZE - 1 - pred_hv[:, 1]
    preds_lb.append(pred_hv)

    mean_lb = np.mean(np.stack(preds_lb), axis=0)
    full_points = _to_full_display(mean_lb, crop_box, meta, full_rad.shape)
    snap_seg = label_mask_display if label_mask_display is not None else seg_display
    snapped = _snap_all(full_points, snap_seg.astype(np.float32))

    landmarks = {
        name: (float(snapped[i, 0]), float(snapped[i, 1]))
        for i, name in enumerate(LANDMARK_NAMES)
    }

    return {
        "landmarks": landmarks,
        "crop_box": crop_box,
        "letterbox_meta": meta,
    }
