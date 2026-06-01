# AI Tools Disclosure

As required by the BrailleVision Hackathon 2026 rules, this document discloses all AI tools and vibe-coding workflows used during the development of SPARSH.

---

## AI Tools Used

| Tool | Purpose | Usage |
|------|---------|-------|
| **Kiro (Amazon)** | AI-powered IDE / coding assistant | Primary development assistant — architecture design, code generation, debugging, algorithm refinement |
| **GitHub Copilot** | Inline code completion | Minor autocomplete suggestions during typing |

---

## Scope of AI Assistance

### What AI helped with:
- Initial project scaffolding (FastAPI routes, WebSocket handler, frontend structure)
- OpenCV preprocessing pipeline design (CLAHE, adaptive thresholding, perspective correction)
- Dot detection algorithm (contour filtering, blob detection, gap analysis)
- Cell segmentation logic (bimodal gap distribution analysis)
- Pattern-to-slot assignment algorithm (column/row classification)
- Frontend UI (glassmorphic design, WebSocket client, drag-and-drop upload)
- Debugging and fixing the dot detector (gap threshold tuning, row assignment logic)
- Writing `inference.py`, `setup_instructions.md`, and this disclosure

### What was done manually / by the team:
- Problem analysis and algorithm strategy decisions
- Braille cell pattern definitions (`braille_cells.json`)
- Dataset collection and labeling strategy
- Testing on real physical Braille samples
- Hyperparameter tuning for real-world conditions
- Final integration and submission

---

## Original Work Statement

All code in this repository was written during or in preparation for the hackathon. The core algorithms (bimodal gap segmentation, column/row dot assignment, hybrid KNN classifier) were designed specifically for this project. No code was copied from other projects or submissions.

The AI tools were used as a coding assistant — similar to using Stack Overflow or documentation — to accelerate implementation. All algorithmic decisions, architecture choices, and debugging were driven by the team.

---

## Model

The model (`backend/models/ml_model.npz`) is a KNN classifier trained on:
- Synthetic Braille dot patterns (augmented with noise and distortion)
- The 1,560-image `braille_char_dataset/` (real photographed Braille characters)

No pre-trained neural network weights from external sources were used.
