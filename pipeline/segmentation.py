"""Region 1: femoral head and shaft segmentation."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from pipeline.config import SEG_NUM_CLASSES, SEG_OVERLAP, SEG_PATCH_SIZE
from pipeline.image_utils import make_display_combined_mask, normalize_percentile
from pipeline.model_registry import ModelRegistry


def _keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    labeled, num = connected_components(binary_mask.astype(np.uint8))
    if num == 0:
        return binary_mask.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest).astype(np.uint8)


def _postprocess_prediction(pred_mask: np.ndarray) -> np.ndarray:
    head = _keep_largest_component(pred_mask == 1)
    shaft = _keep_largest_component(pred_mask == 2)
    out = np.zeros(pred_mask.shape, dtype=np.uint8)
    out[head > 0] = 1
    out[(shaft > 0) & (head == 0)] = 2
    return out


def _gaussian_weight(patch_size: int) -> np.ndarray:
    y = np.linspace(-1, 1, patch_size)
    x = np.linspace(-1, 1, patch_size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dist = xx**2 + yy**2
    weight = np.exp(-dist / (2 * 0.5**2)).astype(np.float32)
    return weight / weight.max()


@torch.no_grad()
def run_segmentation(image: np.ndarray, registry: ModelRegistry | None = None) -> dict:
    """
    Run sliding-window femur segmentation on a NIfTI-space grayscale array (X×Y).

    Returns label mask (0/1/2), display mask (0/128/255), and component masks
    in NIfTI storage coordinates. Call to_display_space() before landmark stages.
    """
    registry = registry or ModelRegistry.get()
    model = registry.segmentation
    device = registry.device

    img_norm = normalize_percentile(image)
    pred = _sliding_window_predict(model, img_norm, device)
    display = make_display_combined_mask(pred)

    return {
        "label_mask": pred,
        "display_mask": display,
        "head_mask": (pred == 1).astype(np.uint8),
        "shaft_mask": (pred == 2).astype(np.uint8),
    }


@torch.no_grad()
def _sliding_window_predict(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    patch_size: int = SEG_PATCH_SIZE,
    overlap: int = SEG_OVERLAP,
) -> np.ndarray:
    original_h, original_w = img.shape
    stride = patch_size - overlap

    pad_h = max(0, patch_size - original_h)
    pad_w = max(0, patch_size - original_w)

    extra_h = (
        math.ceil((original_h + pad_h - patch_size) / stride) * stride + patch_size
    ) - (original_h + pad_h)
    extra_w = (
        math.ceil((original_w + pad_w - patch_size) / stride) * stride + patch_size
    ) - (original_w + pad_w)

    pad_h += extra_h
    pad_w += extra_w
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant")

    h, w = padded.shape
    prob_sum = np.zeros((SEG_NUM_CLASSES, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    weight = _gaussian_weight(patch_size)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = padded[y : y + patch_size, x : x + patch_size]
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            prob_sum[:, y : y + patch_size, x : x + patch_size] += probs * weight[None, :, :]
            weight_sum[y : y + patch_size, x : x + patch_size] += weight

    prob_sum /= np.maximum(weight_sum[None, :, :], 1e-7)
    pred = np.argmax(prob_sum, axis=0).astype(np.uint8)
    pred = pred[:original_h, :original_w]
    return _postprocess_prediction(pred)
