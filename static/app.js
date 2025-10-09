const statusEl = document.querySelector("#status");
const predictionEl = document.querySelector("#prediction");
const confidenceEl = document.querySelector("#confidence");
const speakToggleEl = document.querySelector("#speak-toggle");
const videoEl = document.querySelector("#camera");

const speechSupported = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
let websocket;
let streaming = false;
let speaking = false;
let frameIntervalId;

const formatConfidence = (value) =>
  typeof value === "number" ? `${Math.round(value * 100)}% confidence` : "";

function updateStatus(text, variant = "info") {
  statusEl.textContent = text;
  statusEl.dataset.variant = variant;
}

function speak(text) {
  if (!speechSupported || !speaking || !text) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function handlePrediction(payload) {
  if (!payload || !payload.prediction) {
    predictionEl.textContent = "…";
    confidenceEl.textContent = "Listening";
    return;
  }

  predictionEl.textContent = payload.prediction;
  confidenceEl.textContent = formatConfidence(payload.confidence);
  speak(payload.prediction);
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
    const dataUrl = canvas.toDataURL("image/jpeg", 0.4);
    websocket.send(JSON.stringify({ frame: dataUrl }));
  }, 400);
}

function stopStreaming() {
  streaming = false;
  window.clearInterval(frameIntervalId);
}

function createWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  websocket = new WebSocket(`${protocol}://${window.location.host}/ws`);

  websocket.onopen = () => {
    updateStatus("Connected", "success");
    startStreaming();
  };

  websocket.onclose = () => {
    updateStatus("Disconnected", "warning");
    stopStreaming();
    window.setTimeout(createWebSocket, 2000);
  };

  websocket.onerror = () => {
    updateStatus("Error", "error");
  };

  websocket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.error) {
        updateStatus(`Error: ${payload.error}`, "error");
        return;
      }
      updateStatus("Listening", "success");
      handlePrediction(payload);
    } catch (error) {
      updateStatus("Invalid server response", "error");
    }
  };
}

async function initialiseCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
  } catch (error) {
    updateStatus("Camera access denied", "error");
    throw error;
  }
}

speakToggleEl.addEventListener("click", () => {
  if (!speechSupported) {
    return;
  }
  speaking = !speaking;
  speakToggleEl.textContent = speaking ? "🔊 Voice captions on" : "🔇 Voice captions off";
  if (!speaking) {
    window.speechSynthesis.cancel();
  }
});

if (!speechSupported) {
  speakToggleEl.textContent = "🔇 Voice captions unavailable";
  speakToggleEl.disabled = true;
}

(async () => {
  updateStatus("Awaiting camera permission", "info");
  await initialiseCamera();
  updateStatus("Connecting", "info");
  createWebSocket();
})();
