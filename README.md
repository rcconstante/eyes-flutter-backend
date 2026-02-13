# EYES Vision Backend

FastAPI backend for the EYES assistive mobile application. Processes camera frames through an AI pipeline and returns structured results for spoken/haptic feedback.

## Architecture

```
Camera Frame (JPEG)
        │
        ▼
┌─ Zero-DCE ──────────┐
│ Low-light detection  │ ← Skipped if image is bright enough
│ Image enhancement    │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─ YOLOv8s ─┐  ┌─ MiDaS ──────┐
│ Object     │  │ Depth map    │
│ Detection  │  │ Estimation   │
└─────┬──────┘  └──────┬───────┘
      │                │
      └───────┬────────┘
              ▼
┌─ Post-Processing ────────────┐
│ • Distance mapping           │
│ • Scene classification       │
│ • Currency recognition       │
│ • Priority object selection  │
│ • Safety alerts              │
└──────────────┬───────────────┘
               ▼
         JSON Response
```

## Quick Start (Local)

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Place model weights in models/ (see models/README.md)

# Run server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### `GET /health`
Health check. Returns `{"status": "ok"}`.

### `POST /api/analyze`
Main analysis endpoint.

**Request:** `multipart/form-data` with field `image` (JPEG file).

**Response:**
```json
{
  "priority_object": "car",
  "distance": 2.5,
  "currency": null,
  "scene_type": "Outdoor / Street",
  "alerts": ["car nearby – 2.5m"],
  "detections": [
    {
      "label": "car",
      "confidence": 0.87,
      "bbox": [100, 200, 400, 500],
      "distance": 2.5
    }
  ],
  "enhanced": false,
  "processing_time": 0.342
}
```

## Deploy to Railway

1. Push the `backend/` folder to a GitHub repository (or use the monorepo root).
2. Create a new project on [Railway](https://railway.app).
3. Connect your GitHub repo → set **Root Directory** to `backend`.
4. Railway auto-detects the Python project and uses `nixpacks.toml` + `Procfile`.
5. Set environment variables if needed:
   - `YOLO_MODEL_PATH` (default: `models/yolov8n.pt`)
   - `ZERO_DCE_MODEL_PATH` (default: `models/zero_dce_model.h5`)
   - `CONFIDENCE_THRESHOLD` (default: `0.35`)
6. Upload model weights via Railway volume or include in the repo.

## Models

| Model | Purpose | Source |
|-------|---------|--------|
| **YOLOv8n** | Object detection + currency recognition | Custom-trained (see `yolo_model_training.py`) |
| **Zero-DCE** | Low-light image enhancement | Custom-trained (see `zero_reference_dce.py`) |
| **MiDaS Small** | Monocular depth / distance estimation | Intel ISL (auto-downloaded) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port (set by Railway) |
| `YOLO_MODEL_PATH` | `models/yolov8n.pt` | Path to YOLO weights |
| `ZERO_DCE_MODEL_PATH` | `models/zero_dce_model.h5` | Path to Zero-DCE H5 model |
| `MIDAS_MODEL_TYPE` | `MiDaS_small` | MiDaS variant |
| `CONFIDENCE_THRESHOLD` | `0.35` | YOLO confidence threshold |
| `LOW_LIGHT_THRESHOLD` | `0.35` | Brightness threshold for enhancement |
| `FOCAL_LENGTH_PX` | `500.0` | Approximate focal length for pinhole distance |
