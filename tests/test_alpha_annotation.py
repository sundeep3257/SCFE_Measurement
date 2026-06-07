"""Tests for alpha-angle annotation geometry helpers."""

import math

from pipeline.annotations import _clip_line_through_point, _distal_neck_endpoint
from pipeline.measurements import supplementary_angle_at_vertex


def _acute_arc_span(a1: float, a2: float) -> float:
    """Mirror arc-span logic from _draw_acute_angle_arc for testing."""
    ccw = (a2 - a1) % 360
    if ccw > 180:
        ccw = 360 - ccw
    return ccw if ccw <= 90 else 180.0 - ccw


def test_distal_endpoint_points_toward_shaft_not_head():
    """Distal end lies on the same side of center as the distal shaft landmark."""
    center = (100.0, 80.0)
    distal_shaft = (120.0, 200.0)  # inferior to center (toward shaft)
    proximal_hint = (90.0, 30.0)   # superior to center (toward head)
    neck_direction = (0.2, 1.0)
    w, h = 250, 250

    ep_a, ep_b = _clip_line_through_point(center[0], center[1], neck_direction, w, h)
    distal = _distal_neck_endpoint(center, neck_direction, w, h, distal_shaft)

    assert distal in (ep_a, ep_b)
    assert distal[1] > center[1]  # more inferior than center
    assert distal[1] > proximal_hint[1]


def test_distal_endpoint_opposite_when_shaft_is_superior():
    """If shaft landmark is above center, distal border point is the superior one."""
    center = (100.0, 150.0)
    distal_shaft = (100.0, 50.0)  # shaft landmark superior on this synthetic case
    neck_direction = (0.0, -1.0)
    w, h = 200, 200

    distal = _distal_neck_endpoint(center, neck_direction, w, h, distal_shaft)
    assert distal[1] < center[1]


def test_acute_arc_span_matches_supplementary_measurement():
    """When interior angle is obtuse, arc span equals supplementary (acute) value."""
    center = (0.0, 0.0)
    overlap = (100.0, 0.0)
    distal = (0.0, 100.0)  # 90° interior
    supplementary = supplementary_angle_at_vertex(overlap, center, distal)
    a1 = math.degrees(math.atan2(0, 100))
    a2 = math.degrees(math.atan2(100, 0))
    assert abs(_acute_arc_span(a1, a2) - supplementary) < 0.5

    # Obtuse interior case: rays at 0° and 130°
    overlap2 = (100.0, 0.0)
    distal2 = (
        100.0 * math.cos(math.radians(130)),
        100.0 * math.sin(math.radians(130)),
    )
    sup2 = supplementary_angle_at_vertex(overlap2, center, distal2)
    span2 = _acute_arc_span(0.0, 130.0)
    assert sup2 < 90
    assert abs(span2 - sup2) < 0.5


def test_clip_line_spans_image():
    ep_a, ep_b = _clip_line_through_point(100.0, 100.0, (0.0, 1.0), 200, 300)
    assert abs(ep_a[0] - ep_b[0]) < 1e-3
    assert abs(ep_a[1] - ep_b[1]) > 100
