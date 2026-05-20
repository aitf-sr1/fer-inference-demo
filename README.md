# fer-inference-demo

A web-based demo for real-time facial emotion recognition. Captures webcam frames at a configurable interval, detects the face using MediaPipe BlazeFace, and classifies four student engagement-related emotions using a trained ViT model.

**Emotions detected:** Boredom · Engagement · Confusion · Frustration

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A trained checkpoint (`.pth`) from [fer-vit](https://github.com/aitf-sr1/fer-vit)

---

## Setup

```bash
git clone https://github.com/aitf-sr1/fer-inference-demo
cd fer-inference-demo

uv sync
```

Then place the BlazeFace model in `assets/`:

```bash
mkdir -p assets
curl -L -o assets/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

---

## Adding a model

Drop one or more `.pth` checkpoint files into the `models/` directory. Each checkpoint must have a `config` key embedded (all checkpoints saved by fer-vit do).

```bash
cp /path/to/best_model.pth models/
```

Multiple checkpoints are supported. The settings panel lets you switch between them at runtime without restarting the server.

---

## Running

```bash
uv run uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser.

---

## Project structure

```
fer-inference-demo/
├── app.py                   # FastAPI server
├── fer/
│   ├── vit_model.py         # ViTEmotionModel (vendored from fer-vit)
│   └── auxiliary_encoder.py
├── inference/
│   ├── face_detector.py     # MediaPipe BlazeFace: detect → crop 224×224
│   ├── model_loader.py      # Load checkpoint, run inference
│   └── pipeline.py          # Orchestrates face detection + model inference
├── assets/
│   └── blaze_face_short_range.tflite  # Bundled MediaPipe model
├── models/                  # Place .pth checkpoint files here
├── static/
│   └── index.html           # Frontend (plain HTML/CSS/JS)
└── pyproject.toml
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the frontend |
| `GET` | `/api/models` | Lists available checkpoints and the currently loaded one |
| `POST` | `/api/model` | Load a model by name: `{"model": "best_model.pth"}` |
| `POST` | `/api/infer` | Run inference: `{"image": "<base64>", "model": "best_model.pth"}` |

### Inference response

```json
{
  "face_detected": true,
  "num_classes": 4,
  "emotions": {
    "Boredom": 0,
    "Engagement": 3,
    "Confusion": 1,
    "Frustration": 0
  }
}
```

Each emotion value is a predicted class index. For 4-class models the range is `0–3` (ordinal intensity). For binary models the range is `0–1`.

When no face is detected:

```json
{
  "face_detected": false,
  "num_classes": 4,
  "emotions": null
}
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Directory scanned for `.pth` checkpoint files |
| `MEDIAPIPE_MODEL_PATH` | `assets/blaze_face_short_range.tflite` | Path to the BlazeFace TFLite model |
