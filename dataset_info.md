# Dataset Information — SPARSH Braille Vision

## 1. braille_char_dataset/ (Real Photographed Braille)

| Property | Value |
|----------|-------|
| Total images | 1,560 |
| Image format | JPG |
| Image size | ~28×28 pixels (cropped cells) |
| Classes | 26 (letters a–z) |
| Samples per class | 60 (20 source images × 3 augmentations) |

### Augmentation Types
Each source image has 3 variants:
- `*dim.jpg` — brightness dimmed
- `*rot.jpg` — slight rotation
- `*whs.jpg` — white-space / contrast variant

### Naming Convention
```
{letter}{source_index}{augmentation_type}.jpg
Example: a1.JPG0rot.jpg  →  letter 'a', source image 1, rotation augment
```

### Class Distribution
All 26 letters (a–z) are represented with equal samples.

---

## 2. Synthetic Training Dataset (Generated at Runtime)

| Property | Value |
|----------|-------|
| Source | Programmatically generated from `braille_cells.json` |
| Samples per class | 150 |
| Total samples | ~6,000 |
| Feature format | 6-dimensional float vector |
| Classes | 26 letters + punctuation + contractions |

### Augmentation Applied
- Gaussian noise (σ=0.07) on dot intensity values
- Random dot dropout (18% probability per raised dot)
- Shadow gradient (linear intensity falloff 0.7–0.95)

---

## 3. Braille Pattern Reference (`backend/models/braille_cells.json`)

### Letters (26 patterns)
Standard Grade 1 Braille, 6-bit encoding:
- Bit 0 = dot1 (top-left)
- Bit 1 = dot2 (mid-left)
- Bit 2 = dot3 (bot-left)
- Bit 3 = dot4 (top-right)
- Bit 4 = dot5 (mid-right)
- Bit 5 = dot6 (bot-right)

### Punctuation (12 patterns)
Comma, semicolon, colon, period, exclamation, question mark, apostrophe, quote, hyphen, parenthesis, capital indicator, number indicator.

### Numbers (10 patterns)
Digits 0–9 (same patterns as letters a–j, preceded by number indicator).

### Contractions (5 patterns)
and, for, of, the, with.

---

## 4. Label Data (`label_data/`)

Contains partially labeled real Braille photo data:
- `samples.csv` — detected cell metadata from real photos
- `cells/` — cropped cell preview images
- `annotated/` — annotated source images with detected dots

This data was collected from 20 real Braille screenshot images (`real_braille_photos/`) and can be used to build a real training dataset.

### To build a real dataset:
```bash
# Extract cells from real photos
python backend/scripts/label_tool.py extract \
  --input-dir real_braille_photos \
  --output-dir label_data

# After manually labeling label_data/labels.csv:
python backend/scripts/label_tool.py build \
  --samples label_data/samples.csv \
  --labels label_data/labels.csv \
  --output label_data/real_dataset.npz

# Train with real data
python backend/scripts/train_model.py \
  --dataset label_data/real_dataset.npz \
  --augment-synthetic \
  --output backend/models/ml_model.npz
```

---

## 5. Train/Validation/Test Split

The current KNN model uses all synthetic data for training (no validation split needed for KNN). For future CNN training:

| Split | Ratio | Notes |
|-------|-------|-------|
| Train | 80% | Used for model fitting |
| Validation | 10% | Used for hyperparameter tuning |
| Test | 10% | Final evaluation only |
