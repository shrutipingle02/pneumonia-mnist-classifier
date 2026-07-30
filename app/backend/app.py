"""
PneumoniaScan AI — backend.

Serves the trained PneumoniaMNIST CNN over HTTP so the browser frontend can
use it. The browser cannot run PyTorch, so this sits in between.

Run:  python app.py        ->  http://127.0.0.1:5001
"""

import io
import os
import time
import hashlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))

# The weights live at the repo root, next to train.py that produced them.
# Falling back to a local copy keeps this folder runnable on its own.
MODEL_PATH = next(
    p for p in (
        os.path.join(HERE, "..", "..", "pneumonia_model.pth"),
        os.path.join(HERE, "pneumonia_model.pth"),
    ) if os.path.exists(p)
)

PORT = 5001  # not 5000 — macOS AirPlay Receiver usually holds that port


# --------------------------------------------------------------------------
# Model. Must match train.py exactly or the saved weights will not load.
# --------------------------------------------------------------------------

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device("cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # dropout off — the same X-ray always gives the same answer

N_PARAMS = sum(p.numel() for p in model.parameters())

# Identical preprocessing to training. If this drifts, the model still loads
# but its predictions become meaningless.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Measured on the 624-image PneumoniaMNIST test set. Not estimates.
METRICS = {
    "sensitivity": 99.5,
    "specificity": 56.4,
    "auc": 0.934,
    "test_size": 624,
    "params": N_PARAMS,
}

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify(status="ok", model="PneumoniaCNN", **METRICS)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="No image was uploaded."), 400

    raw = request.files["image"].read()
    if not raw:
        return jsonify(error="The uploaded file was empty."), 400

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        return jsonify(
            error="Unable to read valid image data in the provided file."
        ), 400

    width, height = image.size
    fmt = (image.format or "unknown").upper()

    tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor).squeeze(1)).item()
    inference_ms = (time.perf_counter() - start) * 1000

    # Derived from the image bytes, so the same X-ray always gets the same
    # reference. This is a scan reference, not a patient identifier.
    scan_ref = hashlib.sha256(raw).hexdigest()[:6].upper()

    return jsonify(
        probability=probability,
        label="PNEUMONIA" if probability > 0.5 else "NORMAL",
        inference_ms=round(inference_ms, 2),
        scan_ref=scan_ref,
        image_width=width,
        image_height=height,
        image_format=fmt,
        **METRICS,
    )


if __name__ == "__main__":
    print(f"model loaded — PneumoniaCNN, {N_PARAMS:,} parameters")
    print(f"listening on http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
