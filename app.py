"""Flask application for SCFE radiograph measurement."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from pipeline.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, PROJECT_ROOT
from pipeline.image_utils import ImageValidationError, validate_png_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = PROJECT_ROOT / "uploads"
GENERATED_DIR = PROJECT_ROOT / "static" / "generated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.before_request
def _preload_models() -> None:
    """Warm model cache on first request (skipped for health checks)."""
    if app.config.get("TESTING"):
        return
    if request.endpoint in ("health", "static", "index"):
        return
    try:
        from pipeline.model_registry import ModelRegistry

        ModelRegistry.get().ensure_loaded()
    except Exception:
        logger.exception("Model preload failed; will retry on analysis.")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        flash("No image file was provided.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or not file.filename:
        flash("Please select a PNG image to analyze.", "error")
        return redirect(url_for("index"))

    if not _allowed_file(file.filename):
        flash("Only PNG files are supported.", "error")
        return redirect(url_for("index"))

    request_id = uuid.uuid4().hex
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename) or "upload.png"
    if not safe_name.lower().endswith(".png"):
        safe_name = f"{Path(safe_name).stem}.png"

    input_path = request_dir / safe_name
    file.save(input_path)

    try:
        validate_png_upload(input_path)
    except ImageValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    output_dir = GENERATED_DIR / request_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pipeline import PipelineError, run_scfe_pipeline

        result = run_scfe_pipeline(input_path, output_dir)
    except PipelineError as exc:
        logger.warning("Pipeline error for %s: %s", request_id, exc)
        for warning in exc.warnings:
            flash(warning, "warning")
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except Exception:
        logger.exception("Unexpected pipeline failure for %s", request_id)
        flash("An unexpected error occurred during analysis. Please try another image.", "error")
        return redirect(url_for("index"))

    session_data = result.to_dict()
    session_data["request_id"] = request_id
    session_data["upload_filename"] = safe_name

    return render_template(
        "results.html",
        result=session_data,
        request_id=request_id,
    )


@app.route("/results/<request_id>/<filename>")
def serve_result_image(request_id: str, filename: str):
    if ".." in request_id or ".." in filename:
        abort(404)
    directory = GENERATED_DIR / request_id
    if not directory.exists():
        abort(404)
    allowed = {"southwick.png", "alpha.png", "hnor.png"}
    if filename not in allowed:
        abort(404)
    return send_from_directory(directory, filename)


@app.route("/uploads/<request_id>/<filename>")
def serve_upload_preview(request_id: str, filename: str):
    if ".." in request_id or ".." in filename:
        abort(404)
    directory = UPLOAD_DIR / request_id
    if not directory.exists():
        abort(404)
    return send_from_directory(directory, filename)


@app.errorhandler(413)
def too_large(_exc):
    flash(f"File exceeds the maximum upload size of {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Default debug+reloader on for local `python app.py` so code edits take effect.
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
