# Model Information — SPARSH Braille Cell Classifier

## Model File

- **File**: `ml_model.npz`
- **Format**: NumPy compressed archive (`.npz`)
- **Arrays**: `x` (float32, shape N×6) and `y` (int32, shape N)

## Architecture

**TactileKNN** — a hybrid 3-tier classifier:

1. **Exact pattern match** — direct lookup in `braille_cells.json` (26 letters + punctuation + contractions)
2. **Hamming-1 tolerance** — accepts patterns with 1-bit error if geometric confidence ≥ 0.55
3. **KNN fallback** — k=5 nearest neighbors on 6-dimensional intensity feature vectors

## Training

| Parameter | Value |
|-----------|-------|
| Algorithm | K-Nearest Neighbors (k=5) |
| Feature dimension | 6 (one per Braille dot slot) |
| Classes | 26 letters + punctuation + contractions |
| Training samples | ~6,000 (synthetic augmented) |
| Augmentation | Gaussian noise, dot dropout, shadow gradients |

## Training Command

```bash
python backend/scripts/train_model.py \
  --samples-per-class 150 \
  --augment-synthetic \
  --output backend/models/ml_model.npz
```

## Feature Extraction

Each Braille cell is represented as a 6-dimensional float vector:
- Dimensions 0–5 correspond to Braille dots 1–6
- Values are intensity scores (0.0 = dot absent, 1.0 = dot present with full confidence)
- Extracted from the dot detector's column/row assignment algorithm

## Pattern Reference

See `braille_cells.json` for the full 6-bit pattern definitions.

Bit order: `dot1 dot2 dot3 dot4 dot5 dot6`
- dot1 = top-left, dot2 = mid-left, dot3 = bot-left
- dot4 = top-right, dot5 = mid-right, dot6 = bot-right

## Performance

On synthetic test images (clean, well-lit):
- Letter accuracy: ~100%
- Sentence accuracy: ~100%
- Average latency: ~45ms per image

On real photographed Braille (embossed paper):
- Performance depends on image quality, lighting, and camera angle
- Best results with: good lighting, flat paper, camera perpendicular to surface
