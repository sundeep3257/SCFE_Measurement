"""Central configuration for the SCFE measurement pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of pipeline package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths (relative to project root)
MODELS_DIR = PROJECT_ROOT / "models"
SEGMENTATION_MODEL = MODELS_DIR / "best_femur_model.pth"
NECK_LINE_MODEL = MODELS_DIR / "best_femoral_neck_slope_model_roi_crop.pt"
LANDMARK_MODEL = MODELS_DIR / "best_5point_heatmap_unet_boundary_constrained.pt"

# Segmentation (Region 1)
SEG_PATCH_SIZE = 512
SEG_OVERLAP = 128
SEG_NUM_CLASSES = 3

# Neck line (Region 2)
NECK_IMG_SIZE = 256
NECK_ROI_THRESHOLD = 0.1
NECK_PAD_TOP = 20
NECK_PAD_RIGHT = 20
NECK_USE_TTA = True

# Landmarks (Region 3)
LANDMARK_IMG_SIZE = 256
LANDMARK_ROI_THRESHOLD = 0.1
LANDMARK_PAD = 30
LANDMARK_HEAD_MAX = 0.75
LANDMARK_SHAFT_MIN = 0.75

LANDMARK_NAMES = [
    "medial_physis_edge",
    "lateral_physis_edge",
    "anterior_head_peak",
    "anterior_neck_divot",
    "distal_shaft_midpoint",
]

LANDMARK_TO_MASK = {
    "medial_physis_edge": "head",
    "lateral_physis_edge": "head",
    "anterior_head_peak": "head",
    "anterior_neck_divot": "shaft",
    "distal_shaft_midpoint": "shaft",
}

# Display mask encoding (Region 1 output style)
HEAD_DISPLAY_VALUE = 128
SHAFT_DISPLAY_VALUE = 255

# Flask / upload defaults
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "32"))
MIN_IMAGE_DIM = int(os.environ.get("MIN_IMAGE_DIM", "128"))
ALLOWED_EXTENSIONS = {".png"}

# Gunicorn timeout guidance (seconds)
GUNICORN_TIMEOUT = int(os.environ.get("GUNICORN_TIMEOUT", "300"))

# Head circle: dilate fitted radius before overlap detection and downstream calcs
HEAD_CIRCLE_DILATION_FACTOR = 1.2
