"""
Wong colorblind-safe palette (Wong 2011, Nature Methods).

Shared semantic colors used consistently across all annotation images.
"""

from __future__ import annotations

# Wong palette hex → RGBA (alpha 230 for lines, 255 for points)
WONG = {
    "black": (0, 0, 0),
    "orange": (230, 159, 0),
    "sky_blue": (86, 180, 233),
    "green": (0, 158, 115),
    "yellow": (240, 228, 66),
    "blue": (0, 114, 178),
    "vermillion": (213, 94, 0),
    "purple": (204, 121, 167),
}


def _a(rgb: tuple[int, int, int], alpha: int = 230) -> tuple[int, int, int, int]:
    return rgb[0], rgb[1], rgb[2], alpha


# Semantic roles — same color for the same geometry in every image
ANNOTATION_COLORS = {
    "shaft_line": _a(WONG["blue"]),
    "neck_line": _a(WONG["orange"]),
    "physis_line": _a(WONG["green"]),
    "perpendicular_line": _a(WONG["sky_blue"]),
    "landmark": _a(WONG["vermillion"], 255),
    "head_circle": _a(WONG["purple"]),
    "overlap_line": _a(WONG["green"]),
    "overlap_point": _a(WONG["orange"], 255),
    "neck_axis_alpha": _a(WONG["yellow"]),
    "parallel_line_b": _a(WONG["sky_blue"]),
    "distance_marker": _a(WONG["yellow"]),
    "angle_arc": _a(WONG["sky_blue"]),
    "text_bg": (255, 255, 255, 210),
    "text": _a(WONG["black"], 255),
}
