"""
Clinical geometry calculations for SCFE measurements.

All functions use vector-based math with numerically stable angle routines.
Coordinates: x = column (right), y = row (down), origin upper-left.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

Point = Tuple[float, float]
Vector = Tuple[float, float]


class GeometryError(ValueError):
    """Raised when a measurement cannot be computed reliably."""


def _vec(a: Point, b: Point) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-10:
        raise GeometryError("Zero-length direction vector.")
    return v / n


def _clamp_cos(c: float) -> float:
    return max(-1.0, min(1.0, c))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, acute: bool = True) -> float:
    """Return the angle in degrees between two direction vectors."""
    u1 = _unit(v1)
    u2 = _unit(v2)
    cos_theta = _clamp_cos(float(np.dot(u1, u2)))
    degrees = math.degrees(math.acos(cos_theta))
    if acute:
        return min(degrees, 180.0 - degrees)
    return degrees


def angle_at_vertex(p1: Point, vertex: Point, p2: Point) -> float:
    """Angle at vertex formed by segments vertex→p1 and vertex→p2."""
    v1 = _vec(vertex, p1)
    v2 = _vec(vertex, p2)
    u1 = _unit(v1)
    u2 = _unit(v2)
    cos_theta = _clamp_cos(float(np.dot(u1, u2)))
    return math.degrees(math.acos(cos_theta))


def supplementary_angle_at_vertex(p1: Point, vertex: Point, p2: Point) -> float:
    """
    Supplementary intersection angle at vertex (180° minus the interior angle).

    When two lines cross they form two angles; alpha angle uses the supplementary
    angle to the direct vertex measurement.
    """
    interior = angle_at_vertex(p1, vertex, p2)
    return 180.0 - interior


def perpendicular_vector(v: np.ndarray) -> np.ndarray:
    """Rotate direction 90° counter-clockwise: (dx, dy) -> (-dy, dx)."""
    return np.array([-v[1], v[0]], dtype=np.float64)


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def extend_line_through_point(
    point: Point,
    direction: Vector,
    half_length: float,
) -> Tuple[Point, Point]:
    """Return two endpoints of a segment centered on point along direction."""
    d = _unit(np.array(direction, dtype=np.float64))
    cx, cy = point
    return (
        (cx - d[0] * half_length, cy - d[1] * half_length),
        (cx + d[0] * half_length, cy + d[1] * half_length),
    )


def extend_line_segment(a: Point, b: Point, extension: float = 0.25) -> Tuple[Point, Point]:
    """Extend segment beyond endpoints by a fraction of its length."""
    v = _vec(a, b)
    length = _norm(v)
    if length < 1e-10:
        raise GeometryError("Cannot extend a zero-length segment.")
    d = v / length
    ext = length * extension
    return (
        (a[0] - d[0] * ext, a[1] - d[1] * ext),
        (b[0] + d[0] * ext, b[1] + d[1] * ext),
    )


def perpendicular_distance_between_parallel_lines(
    line_direction: Vector,
    point_on_line_a: Point,
    point_on_line_b: Point,
) -> float:
    """
    Perpendicular distance between two parallel lines sharing line_direction,
    passing through point_on_line_a and point_on_line_b respectively.
    """
    dx, dy = line_direction
    mag = math.hypot(dx, dy)
    if mag < 1e-10:
        raise GeometryError("Neck line direction is undefined.")
    return abs(dy * (point_on_line_b[0] - point_on_line_a[0]) - dx * (point_on_line_b[1] - point_on_line_a[1])) / mag


def compute_southwick_angle(
    shaft_start: Point,
    shaft_end: Point,
    medial_physis: Point,
    lateral_physis: Point,
) -> Dict:
    """
    Southwick angle between femoral shaft axis and a line perpendicular to the
    physis line (medial–lateral tips). Returns the acute intersection angle.
    """
    shaft_vec = _vec(shaft_start, shaft_end)
    physis_vec = _vec(medial_physis, lateral_physis)
    perp_vec = perpendicular_vector(physis_vec)

    angle = angle_between_vectors(shaft_vec, perp_vec, acute=True)
    physis_mid = midpoint(medial_physis, lateral_physis)

    return {
        "angle_degrees": angle,
        "physis_midpoint": physis_mid,
        "physis_perpendicular_direction": (float(perp_vec[0]), float(perp_vec[1])),
    }


def compute_alpha_angle(
    circle_center: Point,
    superior_overlap: Point,
    neck_direction: Vector,
) -> Dict:
    """
    Corrected alpha angle: angle at circle center between
    (1) line to superior circle-shaft overlap point, and
    (2) femoral-neck-axis line through center parallel to predicted neck line.
    """
    neck_unit = _unit(np.array(neck_direction, dtype=np.float64))
    neck_axis_point = (
        circle_center[0] + neck_unit[0],
        circle_center[1] + neck_unit[1],
    )
    # Supplementary intersection angle at circle center (not the interior angle).
    angle = supplementary_angle_at_vertex(superior_overlap, circle_center, neck_axis_point)

    half = _norm(_vec(circle_center, superior_overlap)) * 1.2
    neck_seg = extend_line_through_point(circle_center, neck_direction, half)
    overlap_seg = (circle_center, superior_overlap)

    return {
        "angle_degrees": angle,
        "neck_axis_segment": neck_seg,
        "center_to_overlap_segment": overlap_seg,
    }


def compute_head_neck_offset_ratio(
    neck_start: Point,
    neck_end: Point,
    anterior_head_peak: Point,
    anterior_neck_divot: Point,
    head_diameter: float,
) -> Dict:
    """
    Head-neck offset ratio = perpendicular distance between parallel neck-oriented
    lines through anterior head peak and anterior neck divot, divided by head diameter.
    """
    if head_diameter <= 0:
        raise GeometryError("Head diameter must be positive.")

    neck_dir = _vec(neck_start, neck_end)
    distance = perpendicular_distance_between_parallel_lines(
        (float(neck_dir[0]), float(neck_dir[1])),
        anterior_head_peak,
        anterior_neck_divot,
    )
    ratio = distance / head_diameter

    return {
        "ratio": ratio,
        "perpendicular_distance": distance,
        "head_diameter": head_diameter,
    }
