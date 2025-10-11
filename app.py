from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sign_language import (
    PredictionSmoother,
    Settings,
    YOLOSignLanguageModel,
    decode_image,
    get_settings,
)

_LOGGER = logging.getLogger("sign-language-translator")

settings: Settings = get_settings()
model = YOLOSignLanguageModel(settings)

app = FastAPI(title="ASL Realtime Translator", version="0.1.0")

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    if not model.ready:
        _LOGGER.warning(
            "YOLO weights missing. Upload weights to %s or set ASL_MODEL_PATH.",
            settings.model_path,
        )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def healthcheck() -> JSONResponse:
    payload: Dict[str, Any] = {
        "status": "ok",
        "model_available": model.ready,
        "model_path": str(settings.model_path),
    }
    return JSONResponse(payload)


@app.get("/config")
async def client_config() -> JSONResponse:
    payload: Dict[str, Any] = {
        "frame_interval_ms": settings.frame_interval_ms,
        "jpeg_quality": settings.jpeg_quality,
        "confidence_threshold": settings.confidence_threshold,
    }
    return JSONResponse(payload)


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    await socket.accept()
    smoother = PredictionSmoother(settings.smoothing_window)
    loop = asyncio.get_running_loop()

    try:
        while True:
            message = await socket.receive_json()
            frame_data = message.get("frame")
            if not isinstance(frame_data, str):
                await socket.send_json({"error": "invalid_payload"})
                continue

            frame = decode_image(frame_data)
            if frame is None:
                await socket.send_json({"error": "invalid_frame"})
                continue

            detection = await loop.run_in_executor(None, model.predict, frame)
            smoothed = smoother.update(detection)

            if smoothed is None:
                await socket.send_json({"prediction": None})
                continue

            payload = {
                "prediction": smoothed.label,
                "confidence": round(smoothed.confidence, 3),
            }

            if smoothed.bounding_box is not None:
                payload["bbox"] = {
                    "x1": smoothed.bounding_box.x1,
                    "y1": smoothed.bounding_box.y1,
                    "x2": smoothed.bounding_box.x2,
                    "y2": smoothed.bounding_box.y2,
                }

            await socket.send_json(payload)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover - guard rail for unexpected issues
        _LOGGER.exception("WebSocket failure: %s", exc)
        await socket.close(code=1011, reason="Internal server error")
