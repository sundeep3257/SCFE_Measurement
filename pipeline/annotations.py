"""Generate annotated measurement images on the original radiograph."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.colors import ANNOTATION_COLORS as C
from pipeline.image_utils import array_to_display_uint8
from pipeline.measurements import (
    extend_line_segment,
    extend_line_through_point,
    midpoint,
    perpendicular_vector,
)

Point = Tuple[float, float]


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_rgb(base: np.ndarray) -> Image.Image:
    gray = array_to_display_uint8(base)
    return Image.fromarray(gray).convert("RGB")


def _draw_dashed(draw: ImageDraw.ImageDraw, a: Point, b: Point, color, width: int = 2, dash: int = 10):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        x1, y1 = a[0] + ux * pos, a[1] + uy * pos
        x2, y2 = a[0] + ux * end, a[1] + uy * end
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        pos += dash * 2


def _draw_label(draw: ImageDraw.ImageDraw, text: str, xy: Point, font: ImageFont.ImageFont):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=C["text_bg"])
    draw.text((x + pad, y + pad), text, fill=C["text"], font=font)


def _draw_point(draw: ImageDraw.ImageDraw, p: Point, color, radius: int = 5):
    x, y = p
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline=(255, 255, 255, 255),
    )


def _angle_degrees(v: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _draw_interior_angle_arc(
    draw: ImageDraw.ImageDraw,
    center: Point,
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    radius: float,
    color,
):
    """Draw the smaller arc between two direction vectors."""
    a1 = _angle_degrees(v1)
    a2 = _angle_degrees(v2)
    diff = (a2 - a1) % 360
    if diff <= 180:
        start, end = a1, a1 + diff
    else:
        start, end = a2, a2 + (360 - diff)
    try:
        draw.arc(
            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
            start=start,
            end=end,
            fill=color,
            width=2,
        )
    except TypeError:
        pass


def _clip_line_through_point(
    cx: float,
    cy: float,
    direction: Tuple[float, float],
    img_w: int,
    img_h: int,
) -> Tuple[Point, Point]:
    """Clip an infinite line through (cx, cy) to the image rectangle."""
    dx, dy = direction
    mag = math.hypot(dx, dy)
    if mag < 1e-10:
        return (cx, cy), (cx, cy)
    dx, dy = dx / mag, dy / mag

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
        best = (unique[0], unique[1])
        best_dist = -1.0
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best = (unique[i], unique[j])
        return best

    eps = 1.0
    return (
        (min(max(cx - eps, x_min), x_max), min(max(cy - eps, y_min), y_max)),
        (min(max(cx + eps, x_min), x_max), min(max(cy + eps, y_min), y_max)),
    )


def _distal_neck_endpoint(
    center: Point,
    neck_direction: Tuple[float, float],
    img_w: int,
    img_h: int,
    distal_shaft: Point,
) -> Point:
    """
    Distal femoral-neck axis endpoint: image-border intersection on the shaft side
    of the neck line (toward the distal shaft midpoint, away from the head).
    """
    cx, cy = center
    ep_a, ep_b = _clip_line_through_point(cx, cy, neck_direction, img_w, img_h)

    shaft_dx = distal_shaft[0] - cx
    shaft_dy = distal_shaft[1] - cy

    def toward_shaft(ep: Point) -> float:
        vx = ep[0] - cx
        vy = ep[1] - cy
        return vx * shaft_dx + vy * shaft_dy

    return ep_a if toward_shaft(ep_a) >= toward_shaft(ep_b) else ep_b


def _draw_acute_angle_arc(
    draw: ImageDraw.ImageDraw,
    p1: Point,
    vertex: Point,
    p2: Point,
    radius: float,
    color,
):
    """
    Draw the acute intersection-angle arc at vertex between rays vertex→p1 and vertex→p2.

    Matches the reported alpha angle (supplementary to the obtuse interior angle).
    """
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    if math.hypot(*v1) < 1e-6 or math.hypot(*v2) < 1e-6:
        return

    a1 = _angle_degrees(v1)
    a2 = _angle_degrees(v2)
    ccw = (a2 - a1) % 360
    if ccw > 180:
        ccw = 360 - ccw

    # Acute line angle between the two rays (equals displayed alpha when interior is obtuse).
    if ccw <= 90:
        start, end = a1, a1 + ccw
    else:
        acute_span = 180.0 - ccw
        start, end = a2, a2 + acute_span

    try:
        draw.arc(
            [vertex[0] - radius, vertex[1] - radius, vertex[0] + radius, vertex[1] + radius],
            start=start,
            end=end,
            fill=color,
            width=2,
        )
    except TypeError:
        pass


def render_southwick_image(
    background: np.ndarray,
    shaft_line: Dict,
    landmarks: Dict,
    southwick: Dict,
    output_path: Path,
) -> Path:
    img = _to_rgb(background)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font()

    med = landmarks["medial_physis_edge"]
    lat = landmarks["lateral_physis_edge"]
    shaft_ext = extend_line_segment(shaft_line["start"], shaft_line["end"], extension=0.35)

    draw.line([shaft_ext[0], shaft_ext[1]], fill=C["shaft_line"], width=3)
    draw.line([med, lat], fill=C["physis_line"], width=3)

    mid = southwick["physis_midpoint"]
    perp = southwick["physis_perpendicular_direction"]
    perp_len = max(background.shape) * 0.35
    perp_seg = extend_line_through_point(mid, perp, perp_len)
    if perp[1] < 0:
        perp = (-perp[0], -perp[1])
        perp_seg = extend_line_through_point(mid, perp, perp_len)

    draw.line([perp_seg[0], perp_seg[1]], fill=C["perpendicular_line"], width=3)
    _draw_point(draw, med, C["landmark"])
    _draw_point(draw, lat, C["landmark"])
    _draw_point(draw, mid, C["perpendicular_line"], radius=4)

    shaft_vec = (shaft_line["end"][0] - shaft_line["start"][0], shaft_line["end"][1] - shaft_line["start"][1])
    _draw_interior_angle_arc(draw, mid, shaft_vec, perp, 40, C["angle_arc"])
    _draw_label(draw, f"Southwick: {southwick['angle_degrees']:.1f}°", (12, 12), font)

    img.save(output_path, format="PNG", optimize=True)
    return output_path


def render_alpha_image(
    background: np.ndarray,
    circle: Dict,
    alpha: Dict,
    neck_direction: Tuple[float, float],
    landmarks: Dict,
    output_path: Path,
) -> Path:
    img = _to_rgb(background)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font()

    h, w = background.shape
    center = circle["center"]
    cx, cy = center
    r = circle["radius"]

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=C["head_circle"], width=3)
    _draw_point(draw, center, C["landmark"], radius=4)

    overlap = circle["superior_overlap"]
    draw.line([center, overlap], fill=C["overlap_line"], width=3)
    _draw_point(draw, overlap, C["overlap_point"], radius=7)

    distal_shaft = landmarks["distal_shaft_midpoint"]
    distal = _distal_neck_endpoint(center, neck_direction, w, h, distal_shaft)
    draw.line([center, distal], fill=C["neck_axis_alpha"], width=3)

    _draw_acute_angle_arc(
        draw,
        overlap,
        center,
        distal,
        min(r * 0.45, 48),
        C["angle_arc"],
    )
    _draw_label(draw, f"Alpha: {alpha['angle_degrees']:.1f}°", (12, 12), font)

    img.save(output_path, format="PNG", optimize=True)
    return output_path


def render_hnor_image(
    background: np.ndarray,
    neck_direction: Tuple[float, float],
    landmarks: Dict,
    hno: Dict,
    output_path: Path,
) -> Path:
    img = _to_rgb(background)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font()

    head_pt = landmarks["anterior_head_peak"]
    neck_pt = landmarks["anterior_neck_divot"]
    half = max(background.shape) * 0.25

    head_line = extend_line_through_point(head_pt, neck_direction, half)
    neck_line = extend_line_through_point(neck_pt, neck_direction, half)

    draw.line([head_line[0], head_line[1]], fill=C["neck_line"], width=3)
    draw.line([neck_line[0], neck_line[1]], fill=C["parallel_line_b"], width=3)
    _draw_point(draw, head_pt, C["landmark"], radius=6)
    _draw_point(draw, neck_pt, C["landmark"], radius=6)

    dx, dy = neck_direction
    mag = math.hypot(dx, dy)
    if mag > 1e-10:
        px, py = -dy / mag, dx / mag
        connector = (
            (head_pt[0] + px * 5, head_pt[1] + py * 5),
            (neck_pt[0] - px * 5, neck_pt[1] - py * 5),
        )
        _draw_dashed(draw, connector[0], connector[1], C["distance_marker"], width=2)

    _draw_label(draw, f"HNO ratio: {hno['ratio']:.3f}", (12, 12), font)

    img.save(output_path, format="PNG", optimize=True)
    return output_path
