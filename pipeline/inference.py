"""High-level SCFE pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.annotations import render_alpha_image, render_hnor_image, render_southwick_image
from pipeline.head_circle import fit_head_circle
from pipeline.image_utils import (
    ImageValidationError,
    load_nifti_display_2d,
    load_nifti_raw_2d,
    png_to_nifti,
    to_display_space,
)
from pipeline.landmarks import predict_landmarks
from pipeline.measurements import (
    GeometryError,
    compute_alpha_angle,
    compute_head_neck_offset_ratio,
    compute_southwick_angle,
)
from pipeline.model_registry import ModelRegistry, PipelineModelError
from pipeline.neck_line import predict_neck_line
from pipeline.segmentation import run_segmentation
from pipeline.shaft_pca import compute_shaft_line

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """User-facing pipeline failure with a concise message."""

    def __init__(self, message: str, warnings: Optional[List[str]] = None):
        super().__init__(message)
        self.warnings = warnings or []


@dataclass
class PipelineResult:
    southwick_angle: float
    alpha_angle: float
    head_neck_offset_ratio: float
    southwick_image: str
    alpha_image: str
    hnor_image: str
    original_image: str
    landmarks: Dict[str, tuple[float, float]]
    geometry: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "southwick_angle": self.southwick_angle,
            "alpha_angle": self.alpha_angle,
            "head_neck_offset_ratio": self.head_neck_offset_ratio,
            "southwick_image": self.southwick_image,
            "alpha_image": self.alpha_image,
            "hnor_image": self.hnor_image,
            "original_image": self.original_image,
            "landmarks": self.landmarks,
            "geometry": self.geometry,
            "warnings": self.warnings,
            "image_shape": {"height": self.image_shape[0], "width": self.image_shape[1]},
        }


def run_scfe_pipeline(
    input_image_path: str | Path,
    output_directory: str | Path,
    registry: ModelRegistry | None = None,
) -> PipelineResult:
    """
    Run the complete SCFE measurement pipeline on a PNG radiograph.

    The uploaded PNG is first converted to NIfTI using the project's standard
    orientation (transpose to X×Y storage), then each stage loads data with the
    same conventions as Implement_Pipeline.py.
    """
    input_path = Path(input_image_path)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    try:
        registry = registry or ModelRegistry.get()
        registry.ensure_loaded()
    except PipelineModelError as exc:
        raise PipelineError(str(exc)) from exc

    try:
        nifti_path = png_to_nifti(input_path, input_path.with_suffix(".nii"))
        img_nifti = load_nifti_raw_2d(nifti_path)
        img_display = load_nifti_display_2d(nifti_path)
    except ImageValidationError as exc:
        raise PipelineError(str(exc)) from exc

    h, w = img_display.shape
    logger.info(
        "Processing %s → NIfTI %s | nifti=%s display=%s",
        input_path.name,
        nifti_path.name,
        img_nifti.shape,
        img_display.shape,
    )

    try:
        seg = run_segmentation(img_nifti, registry)
    except Exception as exc:
        logger.exception("Segmentation failed")
        raise PipelineError("Femur segmentation failed.") from exc

    display_mask = to_display_space(seg["display_mask"])
    label_mask_display = to_display_space(seg["label_mask"])
    shaft_mask_display = to_display_space(seg["shaft_mask"].astype(np.uint8))
    head_mask_display = to_display_space(seg["head_mask"].astype(np.uint8))

    if head_mask_display.sum() < 10 or shaft_mask_display.sum() < 10:
        raise PipelineError("Segmentation did not detect sufficient femoral head or shaft tissue.")

    try:
        shaft_line = compute_shaft_line(shaft_mask_display)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc

    try:
        neck = predict_neck_line(shaft_mask_display, registry)
    except Exception as exc:
        logger.exception("Neck line prediction failed")
        raise PipelineError("Femoral neck line prediction failed.") from exc

    try:
        lm = predict_landmarks(img_display, display_mask, label_mask_display, registry)
        landmarks = lm["landmarks"]
    except Exception as exc:
        logger.exception("Landmark detection failed: %s", exc)
        raise PipelineError("Anatomical landmark detection failed.") from exc

    try:
        circle = fit_head_circle(display_mask)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc

    try:
        southwick = compute_southwick_angle(
            shaft_line["start"],
            shaft_line["end"],
            landmarks["medial_physis_edge"],
            landmarks["lateral_physis_edge"],
        )
        alpha = compute_alpha_angle(
            circle["center"],
            circle["superior_overlap"],
            neck["direction"],
        )
        hno = compute_head_neck_offset_ratio(
            neck["start"],
            neck["end"],
            landmarks["anterior_head_peak"],
            landmarks["anterior_neck_divot"],
            circle["diameter"],
        )
    except GeometryError as exc:
        raise PipelineError(str(exc)) from exc

    southwick_path = output_dir / "southwick.png"
    alpha_path = output_dir / "alpha.png"
    hnor_path = output_dir / "hnor.png"

    try:
        render_southwick_image(img_display, shaft_line, landmarks, southwick, southwick_path)
        render_alpha_image(img_display, circle, alpha, neck["direction"], landmarks, alpha_path)
        render_hnor_image(img_display, neck["direction"], landmarks, hno, hnor_path)
    except Exception as exc:
        logger.exception("Annotation rendering failed")
        raise PipelineError("Failed to generate annotated images.") from exc

    geometry = {
        "coordinate_system": {
            "origin": "upper-left",
            "x_axis": "column (right)",
            "y_axis": "row (down)",
            "display_shape": {"height": h, "width": w},
            "nifti_shape": {"x": img_nifti.shape[0], "y": img_nifti.shape[1]},
            "png_to_nifti": "transpose PNG (H,W) to NIfTI (X,Y); display = nifti.T",
        },
        "nifti_path": str(nifti_path),
        "shaft_line": shaft_line,
        "neck_line": {
            "start": neck["start"],
            "end": neck["end"],
            "direction": neck["direction"],
            "theta_degrees": float(np.degrees(neck["theta_rad"])),
        },
        "head_circle": circle,
        "southwick": southwick,
        "alpha": alpha,
        "head_neck_offset": hno,
    }

    return PipelineResult(
        southwick_angle=southwick["angle_degrees"],
        alpha_angle=alpha["angle_degrees"],
        head_neck_offset_ratio=hno["ratio"],
        southwick_image=str(southwick_path),
        alpha_image=str(alpha_path),
        hnor_image=str(hnor_path),
        original_image=str(input_path),
        landmarks=landmarks,
        geometry=geometry,
        warnings=warnings,
        image_shape=(h, w),
    )
