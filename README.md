# 🧿 SPARSH — Tactile Intelligence Reader

**Sanskrit / Hindi (स्पर्श)** — *meaning "Touch" or "Tactility"*

> **BrailleVision Hackathon 2026 Submission**

SPARSH is a real-time physical Braille recognition system that converts camera images of embossed or handwritten Braille into English text and synthesized speech. It works on live webcam feeds and static photo uploads, with no internet connection required.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open the web app
# Navigate to: http://localhost:8000/app
```

**Standalone inference (no server needed):**
```bash
python inference.py --source sample_inputs/hello.png
python inference.py --source sample_inputs/ --output-dir sample_outputs/
```

---

## 📋 Project Description

SPARSH addresses the challenge of reading real physical Braille — embossed paper, handwritten Braille cards, and tactile documents — using a standard camera. The system:

1. Captures a frame from a live webcam or accepts an uploaded photo
2. Preprocesses the image (CLAHE contrast enhancement, adaptive thresholding, perspective correction)
3. Detects Braille dots using contour analysis and blob detection
4. Segments dots into 2×3 Braille cells using bimodal gap analysis
5. Classifies each cell using a hybrid exact-match / Hamming-1 / KNN classifier
6. Decodes the cell sequence into English text (Grade 1 Braille)
7. Synthesizes speech output via pyttsx3 (offline) or gTTS (cloud fallback)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Computer Vision | OpenCV 4.x, NumPy |
| ML Classifier | Custom KNN (TactileKNN) |
| Text-to-Speech | pyttsx3 (offline), gTTS (cloud) |
| Frontend | Vanilla JS, HTML5, CSS3 (glassmorphic) |
| Real-time | WebSocket (FastAPI + browser) |
| Storage | Local JSON database + file system |

**Model type**: Hybrid (OpenCV + custom KNN)

---

## 📁 Repository Structure

```
braille-vision/
├── README.md
├── setup_instructions.md
├── ai_tools_disclosure.md
├── inference.py                    ← Standalone inference script
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── backend/
│   ├── app.py                      ← FastAPI server entry point
│   ├── requirements.txt
│   ├── core/
│   │   ├── preprocessing.py        ← CLAHE, thresholding, perspective
│   │   ├── dot_detector.py         ← Dot detection + cell segmentation
│   │   ├── braille_mapper.py       ← Pattern → character decoder
│   │   ├── ml_model.py             ← Hybrid KNN classifier
│   │   ├── tts_engine.py           ← Text-to-speech engine
│   │   ├── recognition.py          ← Pipeline orchestrator
│   │   └── history_db.py           ← Local scan history database
│   ├── models/
│   │   ├── braille_cells.json      ← Grade 1 Braille pattern definitions
│   │   ├── ml_model.npz            ← Trained KNN model weights
│   │   └── model_info.md           ← Model documentation
│   ├── routes/
│   │   ├── upload.py               ← POST /api/upload
│   │   ├── stream.py               ← WS /ws/stream
│   │   └── health.py               ← GET /health
│   └── scripts/
│       ├── train_model.py          ← Model training script
│       ├── test_pipeline.py        ← Pipeline test script
│       ├── label_tool.py           ← Dataset labeling helper
│       └── extract_char_dataset.py ← Character dataset extractor
│
├── sample_inputs/                  ← Sample Braille PNG images
│   ├── hello.png
│   ├── abc.png
│   ├── braille.png
│   ├── good_morning.png
│   ├── i_love_you.png
│   └── generate_samples.py        ← Regenerate sample images
│
├── sample_outputs/                 ← Annotated output images + results.json
│
└── braille_char_dataset/           ← 1,560 real Braille character images
```

---

## 🧠 How to Run Locally

### Option A: Full Web App

```bash
cd backend
pip install -r requirements.txt
python app.py
# Open http://localhost:8000/app
```

### Option B: Standalone Inference Script

```bash
# Single image
python inference.py --source sample_inputs/hello.png

# Folder of images
python inference.py --source sample_inputs/ --output-dir sample_outputs/

# With explicit model weights
python inference.py --source sample_inputs/hello.png --weights backend/models/ml_model.npz
```

### Option C: Test Pipeline

```bash
python backend/scripts/test_pipeline.py
```

---

## 🔍 Inference Command (Judge Testing)

```bash
python inference.py --source sample_inputs/test_braille.jpg --weights backend/models/ml_model.npz
```

---

## 📊 Dataset Details

### Dataset Download

Due to GitHub file-count and storage limitations, the complete dataset is hosted on Google Drive.

Dataset Link:
https://drive.google.com/drive/folders/11GmUOVGc0y7_e8zi2NQoBjvjXFTwgQQE?usp=drive_link

The dataset includes:

* braille_char_dataset
* real_braille_photos

Please download the dataset and place the folders in the project root before running training scripts.

### Primary Dataset: `braille_char_dataset/`

| Property | Value |
|----------|-------|
| Source | Real photographed Braille character images |
| Total images | 1,560 |
| Format | JPG (28×28 cropped cells) |
| Classes | 26 letters (a–z) |
| Augmentations | Rotation, brightness dimming, white-space variants |
| Naming convention | `{letter}{index}{augtype}.jpg` (e.g. `a1.JPG0rot.jpg`) |

### Training Dataset: Synthetic Augmented

| Property | Value |
|----------|-------|
| Source | Programmatically generated from `braille_cells.json` |
| Samples per class | 150 |
| Total samples | ~6,000 |
| Augmentations | Gaussian noise, dot dropout (18%), shadow gradients |
| Feature format | 6-dimensional float vector (one per dot slot) |

### Braille Pattern Reference: `backend/models/braille_cells.json`

- 26 letters (Grade 1)
- Punctuation (comma, period, question mark, etc.)
- Number indicator + digits 0–9
- Common contractions (and, for, of, the, with)

---

## 🤖 Model Details

**File**: `backend/models/ml_model.npz`  
**Format**: NumPy `.npz` (arrays `x` shape N×6, `y` shape N)  
**Architecture**: TactileKNN — 3-tier hybrid classifier

### Classification Pipeline (per cell):
1. **Exact match** → direct lookup in pattern dictionary (confidence ~0.88)
2. **Hamming-1** → accept 1-bit error if geometric confidence ≥ 0.55
3. **KNN (k=5)** → nearest neighbor on 6D intensity vector

### Training Command:
```bash
python backend/scripts/train_model.py \
  --samples-per-class 150 \
  --augment-synthetic \
  --output backend/models/ml_model.npz
```

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | K-Nearest Neighbors (k=5) |
| Feature dimension | 6 |
| Classes | 26 letters + punctuation + contractions |
| Training samples | ~6,000 (synthetic) |
| Epochs | N/A (KNN is non-parametric) |
| Augmentation | Gaussian noise σ=0.07, dot dropout 18%, shadow gradient |

**Training logs**: See `backend/scripts/train_model.py` — training is fast (<1s) and runs at server startup if no saved model is found.

---

## 🔬 Algorithm: How It Works

### 1. Preprocessing (`core/preprocessing.py`)
- Upscale small images to minimum 280px
- Convert to grayscale
- CLAHE contrast enhancement (clipLimit=2.5, tileGrid=8×8)
- Multi-strategy binarization: Otsu, adaptive Gaussian, fixed threshold
- Select best binary variant by dot candidate count
- Optional perspective correction (document corner detection)

### 2. Dot Detection (`core/dot_detector.py`)
- Contour-based blob detection with circularity filter (≥0.38)
- SimpleBlobDetector fallback for missed dots
- Dot merging for duplicate detections
- Line clustering by y-coordinate gap analysis

### 3. Cell Segmentation (`core/dot_detector.py`)
- Sort dots by x-coordinate within each line
- **Bimodal gap analysis**: find the largest relative jump in sorted gaps to separate intra-cell gaps (left/right column) from inter-cell gaps (between letters)
- **Word gap detection**: require ≥2 large gaps AND gap ≥ 4× letter_split to insert space cells
- Assign dots to 6 slots (dot1–dot6) using column/row classification:
  - Left/right column split by x-median
  - Top/mid/bot row assignment by y-position relative to estimated row pitch

### 4. Classification (`core/ml_model.py`)
- Extract 6-bit binary pattern from slot assignments
- Exact match → Hamming-1 → KNN cascade
- Blend geometric confidence with classifier confidence
- Return character + confidence percentage

### 5. Decoding (`core/braille_mapper.py`)
- Handle capital indicator (000001) and number indicator (010011)
- Map patterns to Grade 1 characters
- Reconstruct full text with proper spacing

---

## 🎥 Real-Time Performance

| Metric | Value |
|--------|-------|
| WebSocket frame rate | Up to 15 fps |
| Processing latency | ~30–80ms per frame |
| Frame resolution sent | 640px wide (downsampled) |
| TTS latency | <200ms (browser speech synthesis) |

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + version info |
| `/api/upload` | POST | Upload image → decoded text + annotated image |
| `/ws/stream` | WebSocket | Real-time frame streaming |
| `/api/history` | GET | Retrieve scan history |
| `/api/history` | DELETE | Clear scan history |
| `/api/braille/reference` | GET | Full pattern reference table |
| `/docs` | GET | Swagger interactive API docs |

---

## 📦 Sample Inputs & Outputs

Sample input images are in `sample_inputs/` (PNG format, synthetic Braille).

To regenerate:
```bash
python sample_inputs/generate_samples.py
```

Sample outputs (annotated images + `results.json`) are in `sample_outputs/`.

---

## 🔊 Text-to-Speech

- **Primary**: `pyttsx3` — fully offline, no API key needed
- **Fallback**: `gTTS` — cloud-based, higher quality audio
- **Browser**: Web Speech API used in frontend for instant feedback

---

## 📝 Submission Checklist

- [x] Complete source code (frontend + backend + inference)
- [x] `README.md` with full documentation
- [x] `requirements.txt`
- [x] `setup_instructions.md`
- [x] `inference.py` — standalone inference script
- [x] `sample_inputs/` — sample Braille images
- [x] `sample_outputs/` — annotated output images + `results.json`
- [x] `backend/models/ml_model.npz` — trained model weights
- [x] `backend/models/model_info.md` — model documentation
- [x] `backend/scripts/train_model.py` — training code
- [x] `braille_char_dataset/` — 1,560 real Braille character images
- [x] `ai_tools_disclosure.md` — AI tools used
- [x] Works on real physical Braille inputs (embossed + handwritten)
- [x] Real-time WebSocket streaming at up to 15fps
- [x] Line-wise segmentation (handles multi-line pages)
- [x] Dataset download link provided in README
---

## 🗺️ Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Dot detection + cell segmentation | ✅ Done |
| 2 | Pattern classification + confidence | ✅ Done |
| 3 | Real-time WebSocket streaming | ✅ Done |
| 4 | Frontend UI + history database | ✅ Done |
| 5 | Bimodal gap segmentation fix | ✅ Done |
| 6 | Standalone inference script | ✅ Done |
| 7 | CNN on real labeled dataset | 🔜 Next |
| 8 | Grade 2 Braille contractions | 🔜 Later |

---

## 🤝 AI Tools Disclosure

See [`ai_tools_disclosure.md`](ai_tools_disclosure.md) for full details.

**Tools used**: Kiro (Amazon AI IDE), GitHub Copilot
