# Setup Instructions — SPARSH Tactile Intelligence Reader

## Prerequisites

- **Python 3.9 – 3.12** (recommended: 3.11)
- A modern web browser (Chrome / Edge / Firefox) with camera permissions
- A webcam or phone camera (for live scanning)

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-team>/braille-vision.git
cd braille-vision
```

---

## 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `fastapi` + `uvicorn` — web server
- `opencv-python` — image processing
- `numpy` — numerical operations
- `pyttsx3` + `gTTS` — text-to-speech
- `pydantic`, `python-multipart`, `websockets`

---

## 3. Run the Backend Server

```bash
# From the backend/ directory:
python app.py
```

Or from the project root:

```bash
python backend/app.py
```

The server starts at **http://localhost:8000**

---

## 4. Open the Web Application

Navigate to: **http://localhost:8000/app**

- **Scan tab** — live webcam Braille scanner (WebSocket streaming)
- **Upload tab** — upload a photo of Braille for analysis
- **Codex tab** — Grade 1 Braille reference chart

---

## 5. Run Standalone Inference (No Server Required)

```bash
# Single image
python inference.py --source sample_inputs/hello.png

# Entire folder
python inference.py --source sample_inputs/ --output-dir sample_outputs/

# With explicit model weights
python inference.py --source sample_inputs/ --weights backend/models/ml_model.npz
```

---

## 6. Run the Test Pipeline

```bash
python backend/scripts/test_pipeline.py
```

Expected output:
```
=== Full sentence test ===
Expected: you can do it !
Decoded:  you can do it !
Cells: 15  Confidence: 0.88

=== 'this is a great city' (strip) ===
Expected: this is a great city
Decoded:  this is a great city
...
```

---

## 7. Regenerate Sample Inputs

```bash
python sample_inputs/generate_samples.py
```

This creates synthetic Braille PNG images in `sample_inputs/`.

---

## 8. Train the Model (Optional)

```bash
# Synthetic-only training (default at startup):
python backend/scripts/train_model.py --output backend/models/ml_model.npz

# With real labeled data:
python backend/scripts/train_model.py \
  --dataset label_data/real_dataset.npz \
  --augment-synthetic \
  --output backend/models/ml_model.npz
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/api/upload` | Upload Braille image → decoded text |
| WS | `/ws/stream` | Real-time WebSocket frame stream |
| GET | `/api/history` | Retrieve scan history |
| GET | `/api/braille/reference` | Full Braille pattern reference |
| GET | `/docs` | Interactive Swagger API docs |

---

## Troubleshooting

**Camera not working?**
- Open `http://localhost:8000/app` (not `file://`)
- Allow camera in browser prompt
- Close Zoom/Teams if camera is busy

**Model not loading?**
- Ensure `backend/models/ml_model.npz` exists
- Run `python backend/scripts/train_model.py` to regenerate

**Port already in use?**
- Change port: `uvicorn app:app --port 8001` from `backend/`
