"""Region 2: femoral neck line orientation prediction."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from pipeline.config import (
    NECK_IMG_SIZE,
    NECK_PAD_RIGHT,
    NECK_PAD_TOP,
    NECK_ROI_THRESHOLD,
    NECK_USE_TTA,
)
from pipeline.coordinates import crop_to_full
from pipeline.image_utils import compute_roi_crop_box, crop_array, letterbox_resize, normalize_minmax
from pipeline.model_registry import ModelRegistry

Point = Tuple[float, float]


def _theta_from_target(vec: np.ndarray) -> float:
    return 0.5 * math.atan2(float(vec[1]), float(vec[0]))


def _mask_centroid(arr: np.ndarray) -> Point:
    mask = normalize_minmax(arr) > NECK_ROI_THRESHOLD
    ys, xs = np.where(mask)
    h, w = arr.shape
    if len(xs) == 0:
        return w / 2.0, h / 2.0
    return float(np.mean(xs)), float(np.mean(ys))


def _clipped_line_through_point(
    cx: float, cy: float, theta: float, img_w: int, img_h: int
) -> Tuple[Point, Point]:
    dx = math.cos(theta)
    dy = math.sin(theta)
    x_min, x_max = 0.0, float(img_w - 1)
    y_min, y_max = 0.0, float(img_h - 1)
    points: list[Point] = []

    if abs(dx) > 1e-8:
        for x in (x_min, x_max):
            t = (x - cx) / dx
            y = cy + t * dy
            if y_min - 1e-6 <= y <= y_max + 1e-6:
                points.append((x, min(max(y, y_min), y_max)))

    if abs(dy) > 1e-8:
        for y in (y_min, y_max):
            t = (y - cy) / dy
            x = cx + t * dx
            if x_min - 1e-6 <= x <= x_max + 1e-6:
                points.append((min(max(x, x_min), x_max), y))

    unique: list[Point] = []
    for p in points:
        if not any(abs(p[0] - q[0]) < 1e-5 and abs(p[1] - q[1]) < 1e-5 for q in unique):
            unique.append(p)

    if len(unique) >= 2:
        best_pair = (unique[0], unique[1])
        max_dist = -1.0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (unique[i], unique[j])
        return best_pair

    eps = 1.0
    return (
        (min(max(cx - eps, x_min), x_max), min(max(cy - eps, y_min), y_max)),
        (min(max(cx + eps, x_min), x_max), min(max(cy + eps, y_min), y_max)),
    )


def _preprocess_tensor(crop_arr: np.ndarray, device: torch.device) -> torch.Tensor:
    crop_arr = normalize_minmax(crop_arr)
    letterboxed, _ = letterbox_resize(crop_arr, NECK_IMG_SIZE)
    pil = Image.fromarray((letterboxed * 255).astype(np.uint8))
    return TF.to_tensor(pil).unsqueeze(0).to(device)


@torch.no_grad()
def _predict_vec(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    out = model(tensor)[0].detach().cpu().numpy()
    return out / (np.linalg.norm(out) + 1e-8)


@torch.no_grad()
def _predict_theta(model: torch.nn.Module, crop_arr: np.ndarray, device: torch.device) -> float:
    base = _preprocess_tensor(crop_arr, device)
    if not NECK_USE_TTA:
        return _theta_from_target(_predict_vec(model, base))

    vectors = [_predict_vec(model, base)]
    tensor_h = TF.hflip(base.squeeze(0)).unsqueeze(0)
    vec_h = _predict_vec(model, tensor_h)
    vectors.append(np.array([vec_h[0], -vec_h[1]], dtype=np.float32))

    tensor_v = TF.vflip(base.squeeze(0)).unsqueeze(0)
    vec_v = _predict_vec(model, tensor_v)
    vectors.append(np.array([vec_v[0], -vec_v[1]], dtype=np.float32))

    tensor_hv = TF.vflip(TF.hflip(base.squeeze(0))).unsqueeze(0)
    vec_hv = _predict_vec(model, tensor_hv)
    vectors.append(np.array([vec_hv[0], vec_hv[1]], dtype=np.float32))

    mean_vec = np.mean(np.stack(vectors), axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    return _theta_from_target(mean_vec)


@torch.no_grad()
def predict_neck_line(
    shaft_mask_display: np.ndarray,
    registry: ModelRegistry | None = None,
) -> Dict:
    """
    Predict femoral neck line from a shaft mask in display-space coordinates.

    Matches Implement_Pipeline.py Region 2, which loads shaft NIfTI with .T.
    """
    registry = registry or ModelRegistry.get()
    model = registry.neck_line
    device = registry.device

    full_arr = normalize_minmax(shaft_mask_display.astype(np.float32))

    crop_box = compute_roi_crop_box(
        full_arr,
        threshold=NECK_ROI_THRESHOLD,
        pad_top=NECK_PAD_TOP,
        pad_right=NECK_PAD_RIGHT,
    )
    crop_arr = normalize_minmax(crop_array(full_arr, crop_box))
    crop_h, crop_w = crop_arr.shape

    theta = _predict_theta(model, crop_arr, device)
    cx, cy = _mask_centroid(crop_arr)
    start_crop, end_crop = _clipped_line_through_point(cx, cy, theta, crop_w, crop_h)

    start_full = crop_to_full(start_crop, crop_box)
    end_full = crop_to_full(end_crop, crop_box)

    direction = np.array(
        [end_full[0] - start_full[0], end_full[1] - start_full[1]],
        dtype=np.float64,
    )
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        raise ValueError("Predicted neck line has zero length.")

    return {
        "theta_rad": theta,
        "start": start_full,
        "end": end_full,
        "direction": (float(direction[0] / norm), float(direction[1] / norm)),
        "crop_box": crop_box,
    }
