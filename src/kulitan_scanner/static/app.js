const fileInput = document.getElementById('fileInput');
const uploadScanBtn = document.getElementById('uploadScanBtn');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureScanBtn = document.getElementById('captureScanBtn');
const video = document.getElementById('camera');
const canvas = document.getElementById('captureCanvas');

const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const reasonText = document.getElementById('reasonText');
const topkText = document.getElementById('topkText');

let cameraStream = null;

function setLoading(isLoading) {
  uploadScanBtn.disabled = isLoading;
  captureScanBtn.disabled = isLoading;
  uploadScanBtn.textContent = isLoading ? 'Scanning...' : 'Scan Uploaded Image';
  captureScanBtn.textContent = isLoading ? 'Scanning...' : 'Capture And Scan';
}

function renderResult(result) {
  predictionText.textContent = `Prediction: ${result.prediction}`;
  confidenceText.textContent = `Confidence: ${(result.confidence ?? 0).toFixed(3)}`;
  reasonText.textContent = result.reason || '';

  const topk = (result.top_k || []).map((item, i) => `${i + 1}. ${item.label} (${item.confidence.toFixed(3)})`);
  topkText.textContent = topk.length ? `Top candidates\n${topk.join('\n')}` : '';
}

async function scanWithFormData(formData) {
  setLoading(true);
  try {
    const response = await fetch('/api/scan', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Scan failed');
    }
    renderResult(data);
  } catch (error) {
    reasonText.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

async function scanWithDataUrl(dataUrl) {
  setLoading(true);
  try {
    const response = await fetch('/api/scan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_data_url: dataUrl }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Scan failed');
    }
    renderResult(data);
  } catch (error) {
    reasonText.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

uploadScanBtn.addEventListener('click', async () => {
  if (!fileInput.files || fileInput.files.length === 0) {
    reasonText.textContent = 'Choose an image file first.';
    return;
  }
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  await scanWithFormData(formData);
});

startCameraBtn.addEventListener('click', async () => {
  try {
    if (cameraStream) {
      return;
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = cameraStream;
    reasonText.textContent = '';
  } catch (error) {
    reasonText.textContent = 'Camera access failed: ' + error.message;
  }
});

captureScanBtn.addEventListener('click', async () => {
  if (!video.srcObject) {
    reasonText.textContent = 'Start camera first.';
    return;
  }
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);
  const dataUrl = canvas.toDataURL('image/png');
  await scanWithDataUrl(dataUrl);
});
