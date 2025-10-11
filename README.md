# LumenSign: ASL Realtime Translator

A minimalist end-to-end prototype that streams webcam frames from the browser to a FastAPI backend, performs American Sign Language (ASL) detection with the latest Ultralytics YOLO models, and renders live captions (plus optional speech) on the page.

## Features
- FastAPI backend with WebSocket endpoint for low-latency frame ingestion and responses.
- YOLO-based inference pipeline with configurable confidence threshold and temporal smoothing to stabilise predictions.
- Minimal web UI that captures webcam video, displays predicted glosses, and can read captions aloud via the browser's SpeechSynthesis API.
- Health endpoint and environment-driven configuration (model path, smoothing window, confidence).

## Project layout
```
app.py                  # FastAPI application entry point
sign_language/          # Inference utilities (configuration, model wrapper, smoothing, decoding)
static/                 # Web UI assets
requirements.txt        # Python dependencies
```

## Prerequisites
- Python 3.10+
- Ultralytics YOLO weights trained on ASL signs (recommended to fine-tune YOLOv8/YOLOv10 on a labelled ASL dataset).
- A modern browser that supports `WebSocket`, `getUserMedia`, and `SpeechSynthesis` (Chrome, Edge, Firefox, Safari ≥ 15).

> **Model weights**
>
> Place your trained weights at `models/asl-sign-detector.pt` or set the environment variable `ASL_MODEL_PATH=/path/to/weights.pt`. The repository does not ship a pretrained model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optionally create a `.env` file to override defaults:

```env
ASL_MODEL_PATH=/absolute/path/to/your/asl-weights.pt
ASL_CONFIDENCE_THRESHOLD=0.4
ASL_SMOOTHING_WINDOW=7
ASL_FRAME_INTERVAL_MS=400
ASL_JPEG_QUALITY=0.4
ASL_NO_DETECTION_DELAY_MS=5000
```

## Train the ASL model

Fine-tuning weights for the bundled ASL letters dataset can be done directly from this project:

```bash
python -m sign_language.training.asl_letters.trainer --epochs 50 --batch-size 16 --device <mps|cpu>
```

The script loads the pretrained checkpoint at `sign_language/models/yolo11n.pt`, trains on the images listed in `sign_language/training/asl_letters/data.yaml`, and copies the best weights to `models/asl-sign-detector.pt`. Adjust flags such as `--device`, `--epochs`, `--image-size`, or `--output` if you wish to customise the run. Ultralytics will store full training artifacts under `sign_language/training/runs/<run-name>/`.

### Baseline benchmarks

To compare YOLO11 against alternative pipelines, full instructions and comparison guidance live in `docs/model_comparison.md`.

### Helper CLI

A convenience wrapper keeps the common commands in one place:

```bash
# Start the FastAPI server
./manage.sh serve --reload

# Fine-tune the ASL letters model
./manage.sh train --epochs 50 --batch-size 16 --device <mps|cpu>

# Remove training artifacts (runs directory + generated weights)
./manage.sh clean
```

Pass any additional options after the command and they will be forwarded to uvicorn or the training module.

## Run the translator

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser, allow camera access, and begin signing within the frame. Detected signs appear as captions. Toggle the **Voice captions** button to hear recognised text via your browser's speech synthesis voice.

## Notes & next steps
- YOLO inference runs in a worker thread to keep the FastAPI event loop responsive. Adjust the frame interval in `static/app.js` if you need faster/slower updates.
- For production use, consider batching frames, adding authentication, and exposing bounding box overlays back to the client.
- Extend `sign_language/model.py` to map detected glosses to natural language phrases or trigger downstream actions.
