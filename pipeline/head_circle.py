"""Region 5: femoral head circle approximation and superior overlap point."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from pipeline.config import HEAD_CIRCLE_DILATION_FACTOR

Point = Tuple[float, float]


def _get_masks(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.unique(data[data > 0])
    if len(vals) < 2:
        raise ValueError("Combined segmentation must contain head and shaft labels.")
    return data == vals[0], data == vals[-1]


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No head contour found for circle fitting.")
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def _circle_from_3_points(p1, p2, p3) -> Optional[tuple[float, float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2
    cd = (temp - x3**2 - y3**2) / 2
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-10:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return float(cx), float(cy), float(r)


def _least_squares_circle(points: np.ndarray) -> tuple[float, float, float]:
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones(len(points))])
    b = x**2 + y**2
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    r = np.sqrt(c + cx**2 + cy**2)
    return float(cx), float(cy), float(r)


def _ransac_circle(points: np.ndarray, n_iterations: int = 3000) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    best_inliers = None
    best_score = -np.inf

    for _ in range(n_iterations):
        idx = rng.choice(len(points), 3, replace=False)
        model = _circle_from_3_points(points[idx[0]], points[idx[1]], points[idx[2]])
        if model is None:
            continue
        cx, cy, r = model
        if not np.isfinite(r) or r <= 0:
            continue
        residuals = np.abs(np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) - r)
        inliers = residuals < 2.5
        if np.sum(inliers) < 30:
            continue
        score = np.sum(inliers) - np.mean(residuals[inliers])
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        return _least_squares_circle(points)
    return _least_squares_circle(points[best_inliers])


def _superior_overlap(
    cx: float, cy: float, r: float, shaft_mask: np.ndarray
) -> Point:
    h, w = shaft_mask.shape
    theta = np.linspace(0, 2 * np.pi, 7200, endpoint=False)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xs, ys, xi, yi = xs[valid], ys[valid], xi[valid], yi[valid]
    overlap = shaft_mask[yi, xi]
    pts = np.column_stack([xs[overlap], ys[overlap]])
    if len(pts) == 0:
        raise ValueError("No superior circle-shaft overlap point found.")
    idx = int(np.argmin(pts[:, 1]))
    return float(pts[idx, 0]), float(pts[idx, 1])


def fit_head_circle(seg_display: np.ndarray) -> Dict:
    """
    Fit femoral head circle in display-space coordinates.

    Matches Implement_Pipeline.py Region 5, which loads combined NIfTI with .T.
    """
    data = seg_display.astype(np.uint8)
    head_mask, shaft_mask = _get_masks(data)
    boundary = _boundary_points(head_mask)
    shaft_dilated = binary_dilation(shaft_mask, iterations=3)

    filtered = np.array(
        [
            [x, y]
            for x, y in boundary
            if 0 <= y < shaft_dilated.shape[0]
            and 0 <= x < shaft_dilated.shape[1]
            and not shaft_dilated[int(y), int(x)]
        ],
        dtype=np.float64,
    )
    if len(filtered) < 30:
        filtered = boundary.astype(np.float64)

    cx, cy, r_fitted = _ransac_circle(filtered)
    if not np.isfinite(r_fitted) or r_fitted <= 1.0:
        raise ValueError("Degenerate femoral head circle fit.")

    r_analysis = r_fitted * HEAD_CIRCLE_DILATION_FACTOR
    sx, sy = _superior_overlap(cx, cy, r_analysis, shaft_mask)

    return {
        "center": (float(cx), float(cy)),
        "radius_fitted": float(r_fitted),
        "radius": float(r_analysis),
        "dilation_factor": HEAD_CIRCLE_DILATION_FACTOR,
        "diameter": float(2 * r_analysis),
        "superior_overlap": (float(sx), float(sy)),
    }
