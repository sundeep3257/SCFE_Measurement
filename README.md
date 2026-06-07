# SCFE Measurement Web Application

Flask web application for automated SCFE radiographic measurements on hemipelvis PNG radiographs. The pipeline segments the proximal femur, predicts the femoral neck line and five landmarks, fits a femoral head circle, and computes:

- **Southwick angle** (degrees)
- **Alpha angle** (degrees, corrected femoral-neck-axis method)
- **Head-neck offset ratio**

## Requirements

- Python 3.10 or 3.11 recommended
- Model weights in `models/`:
  - `best_femur_model.pth`
  - `best_femoral_neck_slope_model_roi_crop.pt`
  - `best_5point_heatmap_unet_boundary_constrained.pt`

## Local installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Local launch

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | HTTP port |
| `MAX_UPLOAD_MB` | `32` | Maximum upload size |
| `MIN_IMAGE_DIM` | `128` | Minimum width/height in pixels |
| `SECRET_KEY` | dev default | Flask session secret |
| `GUNICORN_TIMEOUT` | `300` | Documented inference timeout (seconds) |

## Render deployment

**Build command:**

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300
```

Or use the included `Procfile`.

Configure:

- **Health check path:** `/health`
- **Instance type:** at least 2 GB RAM recommended for PyTorch CPU inference
- **Persistent disk:** not required; uploads and generated images are ephemeral per request

> **Note:** Render’s filesystem is ephemeral. Generated images persist only long enough for the results page and download links in the same session.

## Running tests

```bash
pytest tests/ -v
```

Geometry tests run without ML models. Smoke tests cover routes only.

## Project structure

```
app.py                  Flask routes
pipeline/
  inference.py          run_scfe_pipeline()
  measurements.py       Southwick, alpha, HNO geometry
  annotations.py        Annotated PNG generation
  segmentation.py         Region 1
  neck_line.py            Region 2
  landmarks.py            Region 3
  shaft_pca.py            Region 4
  head_circle.py          Region 5
  coordinates.py          Coordinate transforms
  image_utils.py          PNG loading and preprocessing
  model_registry.py       Cached model loading
models/                   Trained weights (not in git if large)
templates/                HTML templates
static/                   CSS, JS, generated outputs
uploads/                  Per-request uploads
Implement_Pipeline.py     Original monolithic pipeline (reference)
Short_SCFE.R              Original R measurement scripts (reference)
```

## Pipeline API

```python
from pipeline import run_scfe_pipeline

result = run_scfe_pipeline("path/to/image.png", "output/dir")
print(result.southwick_angle, result.alpha_angle, result.head_neck_offset_ratio)
```

Uploaded PNGs are automatically converted to NIfTI (transpose to X×Y storage, matching
the batch preparation script) before inference. Segmentation runs in NIfTI space;
landmarks, neck line, and circle fitting run in display space (`nifti.T`).

## Research disclaimer

This application is intended for **research and clinical decision support**. It is not a substitute for qualified clinician interpretation.
