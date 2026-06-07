"""Region 4: femoral shaft principal-component axis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Point = Tuple[float, float]


def compute_shaft_line(shaft_mask: np.ndarray) -> Dict:
    """
    Compute the first principal component of the shaft mask in canonical coordinates.

    Returns start/end endpoints, centroid, and unit direction (x=column, y=row).
    """
    mask = shaft_mask > 0
    if np.sum(mask) < 2:
        raise ValueError("Not enough shaft pixels to compute the femoral shaft axis.")

    rows, cols = np.where(mask)
    coords = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / np.linalg.norm(direction)

    projections = centered @ direction
    min_proj, max_proj = projections.min(), projections.max()

    start = centroid + min_proj * direction
    end = centroid + max_proj * direction

    return {
        "start": (float(start[0]), float(start[1])),
        "end": (float(end[0]), float(end[1])),
        "centroid": (float(centroid[0]), float(centroid[1])),
        "direction": (float(direction[0]), float(direction[1])),
    }
