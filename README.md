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
> The repo ships with two pretrained checkpoints under `models/`: `asl-sign-detector.v2.e50.pt` (letters) and `asl-words-detector.v2.e10.pt` (words). Point `ASL_MODEL_PATH` to whichever file you want to serve (letters by default), or drop in your own weights at `models/asl-sign-detector.pt` / `.pt` of your choice.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optionally create a `.env` file to override defaults (you can copy `example.env` as a starting point):

| Variable | Description |
|----------|-------------|
| `ASL_MODEL_PATH` | Filesystem path to the YOLO checkpoint the backend loads. Use `models/asl-sign-detector.v2.e50.pt` (letters) or swap in your own `.pt`. |
| `ASL_CONFIDENCE_THRESHOLD` | Minimum detection confidence (0–1). Raise to cut false positives, lower to recover borderline detections. |
| `ASL_SMOOTHING_WINDOW` | Number of past predictions used for temporal smoothing; larger windows reduce flicker but add latency. |
| `ASL_FRAME_INTERVAL_MS` | Delay between frames sent from the browser (ms). Increase for slower machines/bandwidth, decrease for faster updates. |
| `ASL_JPEG_QUALITY` | Compression quality (0–1) for browser frame uploads; lower = smaller payloads, higher = sharper hands. |
| `ASL_NO_DETECTION_DELAY_MS` | How long (ms) to wait after the last detection before clearing the caption / inserting a spacer. |

## Train the ASL model

Fine-tuning weights for the bundled ASL datasets can be done directly from this project. Letters (default):

```bash
python -m sign_language.training.asl_letters_v2.trainer --epochs 50 --batch-size 16 --device <mps|cpu>
```

For ASL words:

```bash
python -m sign_language.training.asl_words.trainer --epochs 50 --batch-size 16 --device <mps|cpu>
```

Each script loads the pretrained checkpoint at `sign_language/models/yolo11n.pt`, trains on the images listed in the corresponding `sign_language/training/<dataset>/data.yaml`, and copies the best weights to `models/asl-sign-detector.pt` (letters) or `models/asl-words-detector.pt` (words). Adjust flags such as `--device`, `--epochs`, `--image-size`, or `--output` if you wish to customise the run. Ultralytics stores full training artifacts under `sign_language/training/runs/<run-name>/`.

### Baseline benchmarks

To compare YOLO11 against alternative pipelines, full instructions and comparison guidance live in `docs/model_comparison.md`.

### Helper CLI

A convenience wrapper keeps the common commands in one place:

```bash
# Start the FastAPI server
./manage.sh serve --reload

# Fine-tune the ASL letters model (default dataset)
./manage.sh train --dataset letters --epochs 50 --batch-size 16 --device <mps|cpu>

# Fine-tune the ASL words model
./manage.sh train --dataset words --epochs 50 --batch-size 16 --device <mps|cpu>

# Remove training artifacts (runs directory + generated weights)
./manage.sh clean
```

Pass any additional options after the command and they will be forwarded to uvicorn or the training module.

## Run the translator

```bash
uvicorn app:app --reload
```

or

```bash
./manage.sh serve --reload
```

Open `http://localhost:8000` in your browser, allow camera access, and begin signing within the frame. Detected signs appear as captions. Toggle the **Voice captions** button to hear recognised text via your browser's speech synthesis voice.

### Quick CLI test (no Web UI)

If you just want to sanity-check a trained checkpoint with the Ultralytics CLI, you can run it directly on your webcam feed:

```bash
yolo task=detect mode=predict model=models/asl-words-detector.pt source=0 show=True
```

Replace `asl-words-detector.pt` with any of your exported weights. This command launches the default YOLO window, runs real-time inference from camera index `0`, and overlays detections without needing the FastAPI/web frontend.

## Notes & next steps
- YOLO inference runs in a worker thread to keep the FastAPI event loop responsive. Adjust the frame interval in `static/app.js` if you need faster/slower updates.
