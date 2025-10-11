const statusEl = document.querySelector("#status");
const predictionEl = document.querySelector("#prediction");
const confidenceEl = document.querySelector("#confidence");
const speakToggleEl = document.querySelector("#speak-toggle");
const speakToggleIcon = speakToggleEl?.querySelector(".btn__icon");
const speakToggleLabel = speakToggleEl?.querySelector(".btn__label");
const creditsButton = document.querySelector("#credits-btn");
const clearCaptionsButton = document.querySelector("#clear-captions");
const creditsModal = document.querySelector("#credits-modal");
const closeCreditsBtn = document.querySelector("#close-credits");
const videoEl = document.querySelector("#camera");
const videoFrameEl = document.querySelector(".video-frame");
const overlayCanvas = document.querySelector("#overlay-canvas");
const overlayCtx = overlayCanvas?.getContext("2d");

const speechSupported = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

const LETTER_PRONUNCIATIONS = {
  A: "ayee",
  B: "bee",
  C: "see",
  D: "dee",
  E: "ee",
  F: "eff",
  G: "gee",
  H: "aitch",
  I: "eye",
  J: "jay",
  K: "kay",
  L: "el",
  M: "em",
  N: "en",
  O: "oh",
  P: "pee",
  Q: "cue",
  R: "ar",
  S: "ess",
  T: "tee",
  U: "you",
  V: "vee",
  W: "double you",
  X: "ex",
  Y: "why",
  Z: "zee",
};

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
let websocket;
let streaming = false;
let speaking = false;
let frameIntervalId;
let latestBox = null;
let lastPredictionLabel = null;
let detectionActive = false;
let lastFocusedElement = null;
let frameIntervalMs = 400;
let jpegQuality = 0.4;
const captionTokens = [];
let noDetectionTimeoutId = null;
let noDetectionDelayMs = 5000;
const TAB_TOKEN = " ··· ";

const formatConfidence = (value) =>
  typeof value === "number" ? `${Math.round(value * 100)}% confidence` : "";

function updateStatus(text, variant = "info") {
  statusEl.textContent = text;
  statusEl.dataset.variant = variant;
}

function normaliseSpeechText(text) {
  if (typeof text === "string" && text.length === 1 && LETTER_PRONUNCIATIONS[text]) {
    return LETTER_PRONUNCIATIONS[text];
  }
  return text;
}

function speak(text) {
  if (!speechSupported || !speaking || !text) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(normaliseSpeechText(text));
  utterance.lang = "en-US";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function handlePrediction(payload) {
  if (!payload || !payload.prediction) {
    scheduleNoDetectionSpace();
    updateCaptionDisplay();
    confidenceEl.textContent = "Make a sign";
    drawBoundingBox(null);
    detectionActive = false;
    lastPredictionLabel = null;
    return;
  }

  const { prediction, confidence } = payload;
  clearNoDetectionSpace();
  captionTokens.push(prediction);
  updateCaptionDisplay();
  confidenceEl.textContent = formatConfidence(confidence);
  drawBoundingBox(payload.bbox ?? null);
  const shouldSpeak = !detectionActive || prediction !== lastPredictionLabel;
  if (shouldSpeak) {
    speak(prediction);
  }
  detectionActive = true;
  lastPredictionLabel = prediction;
}

function updateVoiceButton() {
  if (!speakToggleEl || !speakToggleLabel || !speakToggleIcon) {
    return;
  }
  speakToggleLabel.textContent = speaking ? "Voice On" : "Voice Off";
  speakToggleIcon.textContent = speaking ? "🔊" : "🔇";
  speakToggleEl.classList.toggle("active", speaking);
}

function showCreditsModal() {
  if (!creditsModal) {
    return;
  }
  lastFocusedElement = document.activeElement;
  creditsModal.classList.remove("hidden");
  creditsModal.setAttribute("aria-hidden", "false");
  closeCreditsBtn?.focus();
}

function hideCreditsModal() {
  if (!creditsModal) {
    return;
  }
  creditsModal.classList.add("hidden");
  creditsModal.setAttribute("aria-hidden", "true");
  if (lastFocusedElement instanceof HTMLElement) {
    lastFocusedElement.focus();
  }
}

function syncOverlaySize() {
  if (!overlayCanvas || !videoFrameEl) {
    return;
  }

  const { clientWidth, clientHeight } = videoFrameEl;
  if (overlayCanvas.width !== clientWidth || overlayCanvas.height !== clientHeight) {
    overlayCanvas.width = clientWidth;
    overlayCanvas.height = clientHeight;
  }
}

function drawBoundingBox(box) {
  if (!overlayCanvas || !overlayCtx) {
    return;
  }

  latestBox = box;
  syncOverlaySize();

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!box) {
    return;
  }

  const width = overlayCanvas.width;
  const height = overlayCanvas.height;

  const x = box.x1 * width;
  const y = box.y1 * height;
  const w = (box.x2 - box.x1) * width;
  const h = (box.y2 - box.y1) * height;

  overlayCtx.strokeStyle = "rgba(250, 204, 21, 0.95)";
  overlayCtx.lineWidth = Math.max(2, width * 0.01);
  overlayCtx.shadowColor = "rgba(15, 23, 42, 0.8)";
  overlayCtx.shadowBlur = 8;
  overlayCtx.strokeRect(x, y, w, h);
  overlayCtx.shadowBlur = 0;
}

function startStreaming() {
  if (streaming) {
    return;
  }
  streaming = true;
  frameIntervalId = window.setInterval(() => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      return;
    }
    if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }

    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    canvas.width = width;
    canvas.height = height;
    context.drawImage(videoEl, 0, 0, width, height);
    const quality = Math.min(Math.max(jpegQuality, 0.1), 1);
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    websocket.send(JSON.stringify({ frame: dataUrl }));
  }, frameIntervalMs);
}

function stopStreaming() {
  streaming = false;
  window.clearInterval(frameIntervalId);
}

function createWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  websocket = new WebSocket(`${protocol}://${window.location.host}/ws`);

  websocket.onopen = () => {
    updateStatus("CONNECTED", "success");
    startStreaming();
  };

  websocket.onclose = () => {
    updateStatus("DISCONNECTED", "warning");
    stopStreaming();
    window.setTimeout(createWebSocket, 2000);
  };

  websocket.onerror = () => {
    updateStatus("ERROR", "error");
  };

  websocket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.error) {
        updateStatus(`ERROR: ${payload.error}`, "error");
        return;
      }
      updateStatus("CAMERA ON", "success");
      handlePrediction(payload);
    } catch (error) {
      updateStatus("INVALID SERVER RESPONSE", "error");
    }
  };
}

async function initialiseCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
    syncOverlaySize();
  } catch (error) {
    updateStatus("CAMERA ACCESS DENIED", "error");
    throw error;
  }
}

if (overlayCanvas && videoFrameEl && typeof ResizeObserver !== "undefined") {
  const resizeObserver = new ResizeObserver(() => {
    syncOverlaySize();
    if (latestBox) {
      drawBoundingBox(latestBox);
    }
  });
  resizeObserver.observe(videoFrameEl);
}

if (speakToggleEl) {
  speakToggleEl.addEventListener("click", () => {
    if (!speechSupported) {
      return;
    }
    speaking = !speaking;
    updateVoiceButton();
    if (!speaking) {
      window.speechSynthesis.cancel();
    }
  });

  if (!speechSupported) {
    if (speakToggleLabel) {
      speakToggleLabel.textContent = "Unavailable";
    }
    if (speakToggleIcon) {
      speakToggleIcon.textContent = "🚫";
    }
    speakToggleEl.disabled = true;
    speakToggleEl.classList.remove("active");
  } else {
    updateVoiceButton();
  }
}

if (creditsButton) {
  creditsButton.addEventListener("click", showCreditsModal);
}

if (closeCreditsBtn) {
  closeCreditsBtn.addEventListener("click", hideCreditsModal);
}

if (creditsModal) {
  creditsModal.addEventListener("click", (event) => {
    if (event.target === creditsModal) {
      hideCreditsModal();
    }
  });
}

if (clearCaptionsButton) {
  clearCaptionsButton.addEventListener("click", () => {
    captionTokens.length = 0;
    updateCaptionDisplay();
  });
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && creditsModal && !creditsModal.classList.contains("hidden")) {
    hideCreditsModal();
  }
});

async function loadClientConfig() {
  try {
    const response = await fetch("/config");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    if (typeof payload.frame_interval_ms === "number") {
      frameIntervalMs = Math.max(100, Math.floor(payload.frame_interval_ms));
    }
    if (typeof payload.jpeg_quality === "number") {
      jpegQuality = payload.jpeg_quality;
    }
    if (typeof payload.no_detection_delay_ms === "number") {
      noDetectionDelayMs = Math.max(500, Math.floor(payload.no_detection_delay_ms));
    }
  } catch (error) {
    console.warn("Failed to load client config", error);
  }
}

function updateCaptionDisplay() {
  if (!captionTokens.length) {
    predictionEl.textContent = " ··· ";
    return;
  }

  predictionEl.textContent = captionTokens.join(" ");
  while (captionTokens.length > 1 && predictionEl.scrollWidth > predictionEl.clientWidth) {
    captionTokens.shift();
    predictionEl.textContent = captionTokens.join(" ");
  }
}

function scheduleNoDetectionSpace() {
  if (noDetectionTimeoutId !== null) {
    return;
  }
  noDetectionTimeoutId = window.setTimeout(() => {
    if (captionTokens.length !== 0 && captionTokens[captionTokens.length - 1] !== TAB_TOKEN) {
      captionTokens.push(TAB_TOKEN);
      updateCaptionDisplay();
    }
    noDetectionTimeoutId = null;
  }, noDetectionDelayMs);
}

function clearNoDetectionSpace() {
  if (noDetectionTimeoutId !== null) {
    window.clearTimeout(noDetectionTimeoutId);
    noDetectionTimeoutId = null;
  }
}

(async () => {
  updateStatus("AWAITING PERMISSIONS", "info");
  await loadClientConfig();
  await initialiseCamera();
  updateStatus("CONNECTING", "info");
  createWebSocket();
})();
