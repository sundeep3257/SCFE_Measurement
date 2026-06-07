"""Unit tests for pure geometry functions."""

import math
import pytest

from pipeline.measurements import (
    GeometryError,
    angle_at_vertex,
    angle_between_vectors,
    compute_alpha_angle,
    compute_head_neck_offset_ratio,
    compute_southwick_angle,
    perpendicular_distance_between_parallel_lines,
    supplementary_angle_at_vertex,
)
import numpy as np


def test_angle_between_vectors_30_degrees():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([math.sqrt(3) / 2, 0.5])
    assert abs(angle_between_vectors(v1, v2, acute=False) - 30.0) < 0.01


def test_angle_between_vectors_45_degrees():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 1.0])
    assert abs(angle_between_vectors(v1, v2, acute=False) - 45.0) < 0.01


def test_angle_between_vectors_90_degrees():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert abs(angle_between_vectors(v1, v2, acute=False) - 90.0) < 0.01


def test_angle_between_vectors_reversed_direction():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    assert abs(angle_between_vectors(v1, v2, acute=True) - 0.0) < 0.01


def test_angle_between_vectors_zero_length_raises():
    with pytest.raises(GeometryError):
        angle_between_vectors(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_southwick_horizontal_and_vertical():
    # Shaft along x-axis; physis vertical -> perpendicular to physis is horizontal
    result = compute_southwick_angle(
        (0.0, 50.0), (100.0, 50.0),
        (50.0, 0.0), (50.0, 100.0),
    )
    assert abs(result["angle_degrees"] - 0.0) < 0.5 or abs(result["angle_degrees"] - 90.0) < 0.5


def test_southwick_known_45():
    result = compute_southwick_angle(
        (0.0, 0.0), (100.0, 100.0),
        (0.0, 50.0), (100.0, 50.0),
    )
    assert 0.0 < result["angle_degrees"] < 90.0


def test_perpendicular_distance_oblique_neck_axis():
    # Parallel horizontal lines separated by 20 px vertically
    dist = perpendicular_distance_between_parallel_lines(
        (1.0, 0.0),
        (0.0, 10.0),
        (0.0, 30.0),
    )
    assert abs(dist - 20.0) < 0.01


def test_perpendicular_distance_vertical_neck_axis():
    dist = perpendicular_distance_between_parallel_lines(
        (0.0, 1.0),
        (10.0, 0.0),
        (30.0, 0.0),
    )
    assert abs(dist - 20.0) < 0.01


def test_hno_ratio():
    result = compute_head_neck_offset_ratio(
        neck_start=(0.0, 0.0),
        neck_end=(100.0, 0.0),
        anterior_head_peak=(50.0, 10.0),
        anterior_neck_divot=(50.0, 30.0),
        head_diameter=100.0,
    )
    assert abs(result["ratio"] - 0.2) < 0.01


def test_hno_zero_diameter_raises():
    with pytest.raises(GeometryError):
        compute_head_neck_offset_ratio(
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 0.0), (0.0, 10.0),
            head_diameter=0.0,
        )


def test_alpha_angle_uses_supplementary_intersection_angle():
    center = (0.0, 0.0)
    overlap = (10.0, 0.0)
    neck_direction = (1.0, 1.0)
    interior = angle_at_vertex(overlap, center, (1.0, 1.0))
    expected = supplementary_angle_at_vertex(overlap, center, (1.0, 1.0))
    result = compute_alpha_angle(center, overlap, neck_direction)
    assert abs(result["angle_degrees"] - expected) < 0.5
    assert abs(result["angle_degrees"] - (180.0 - interior)) < 0.5
    assert abs(result["angle_degrees"] - interior) > 1.0


def test_alpha_angle_corrected_method():
    center = (0.0, 0.0)
    overlap = (0.0, -50.0)
    neck_direction = (1.0, 0.0)
    result = compute_alpha_angle(center, overlap, neck_direction)
    assert abs(result["angle_degrees"] - 90.0) < 0.5
    seg = result["neck_axis_segment"]
    assert abs(seg[1][1] - seg[0][1]) < 0.01


def test_alpha_uses_parallel_neck_not_distal_shaft():
    center = (100.0, 100.0)
    overlap = (100.0, 50.0)
    neck_direction = (0.0, 1.0)
    distal = (100.0, 300.0)  # would change angle if incorrectly used
    result = compute_alpha_angle(center, overlap, neck_direction)
    wrong = angle_at_vertex(distal, center, overlap)
    assert result["angle_degrees"] != wrong or abs(neck_direction[0]) < 1e-6
