# Braille OCR Structural Audit

## Scope

This audit focused on the live OCR path used by:

- `inference.py`
- `backend/routes/upload.py`
- `backend/routes/stream.py`

The goal was not threshold tuning. The goal was to replace the unstable mixed pipeline with one deterministic execution path that is easier to reason about and debug.

## Active Pipeline After Refactor

All public entry points now flow through one path:

1. `run_recognition()`
2. `analyze_braille_image()`
3. `preprocess_frame()`
4. `detect_dots()`
5. `segment_braille_dots()`
6. `decode_lines_with_metadata()`
7. response payload / text output

Module responsibilities are now separated as follows:

- `backend/core/preprocessing.py`
  - image normalization
  - perspective correction
  - candidate threshold generation
  - deterministic binary selection
- `backend/core/dot_detector.py`
  - connected-component dot extraction
  - merged-blob splitting
  - dot overlay rendering
- `backend/core/segmentation.py`
  - line clustering
  - column clustering
  - cell grouping
  - 2x3 matrix generation
  - pattern encoding
  - segmentation overlay rendering
- `backend/core/braille_mapper.py`
  - pattern lookup
  - stateful text translation for capitals / numbers / spaces
- `backend/core/recognition.py`
  - orchestration only
  - payload shaping
  - debug metadata collection

## Root Causes

### 1. Preprocessing had selectable binaries, but the selector was never used

The old `preprocessing.py` generated multiple threshold variants and defined `select_best_binary()`, but `preprocess_frame()` always returned `variants[0]`.

Effect:

- threshold experimentation existed on paper only
- behavior depended on whichever hardcoded variant happened to be first
- debugging was misleading because the code looked adaptive but was not

### 2. Perspective correction was effectively dead

Two separate issues made it inactive:

- `run_recognition()` always called `preprocess_frame(..., correct_perspective=False)`
- `preprocess_frame()` also hardcoded `use_perspective = False`

Effect:

- API and CLI flags suggested perspective correction existed
- runtime behavior ignored the setting

### 3. Dot detection and segmentation were entangled inside one file

The old `dot_detector.py` mixed:

- raw contour detection
- duplicate merging
- line clustering
- gap heuristics
- per-cell grid fitting
- pattern classification
- overlay generation

Effect:

- one change in geometry logic could silently affect classification
- there was no clean place to inspect just dots, just cells, or just translation

### 4. Multiple competing segmentation strategies coexisted

The old detector contained several conflicting approaches at once:

- `_gap_thresholds()`
- `_find_bimodal_split()`
- `_group_dots_into_cells()`
- `segment_cells_in_line()`
- a commented-out alternate `_pattern_full_grid()`
- a second active `_pattern_full_grid()` with different semantics

Only one branch was actually active, but the inactive logic remained in the same module and implied behavior that never ran.

Effect:

- difficult to tell which heuristic was authoritative
- later edits partially replaced earlier logic without removing it
- the file accumulated dead code and misleading fallback paths

### 5. Cell segmentation used the wrong abstraction

The old pipeline tried to infer letters directly from raw column gaps and then pair columns afterward.

That breaks on valid Braille patterns because:

- some letters use one visible column
- some letters use two visible columns
- consecutive one-column cells can resemble inter-dot spacing
- mixed words produce several legitimate gap scales

Effect:

- neighboring dots and neighboring cells were merged
- clean synthetic images were decoded as the wrong letters
- spaces were inserted or removed inconsistently

### 6. Row assignment inside cells was structurally incorrect

The active `_pattern_full_grid()` in the old detector assigned rows by simple sort order inside each column.

That loses the difference between:

- top + middle
- top + bottom
- middle + bottom

Effect:

- skipped middle rows were frequently interpreted as middle-row dots
- exact pattern generation was unstable even when dots were correctly detected

### 7. Translation had competing decode paths

The old pipeline mixed:

- pattern-based decoding through `decode_lines()`
- per-cell char reconstruction through `decode_from_cell_chars()`

Those paths did not share the same stateful handling for capitals, numbers, and spacing.

Effect:

- the same pattern sequence could be rendered differently depending on which path happened to run
- decoding behavior was harder to verify than it needed to be

### 8. Debug output was noisy but not actionable

The old pipeline printed large amounts of raw contour and pattern data directly to stdout.

Effect:

- logs were difficult to use for diagnosis
- there was no structured summary of which binary was chosen, how many columns were formed, or why spaces were inserted

## Dead Code And Unreachable Behavior Identified

These were structural problems in the original codebase before the refactor:

- `backend/core/preprocessing.py`
  - `select_best_binary()` existed but was never called
  - perspective correction path was unreachable in practice because runtime flags were ignored
- `backend/core/recognition.py`
  - repeated debug `print()` blocks duplicated the same information
  - `TEXT:` logging referenced `full_text` before it was assigned and relied on `if 'full_text' in locals()` to avoid crashing
- `backend/core/dot_detector.py`
  - `_gap_thresholds()` was defined but not used by the active segmentation path
  - `_find_bimodal_split()` was defined but bypassed by hardcoded `letter_split` / `word_split`
  - `_group_dots_into_cells()` was not used by the active detector path
  - the older commented `_pattern_full_grid()` implementation was dead code
  - the active `_pattern_full_grid()` duplicated responsibility with the commented version but behaved differently
  - debug image writes to `debug/` happened unconditionally inside the detector path
- `backend/core/braille_mapper.py`
  - `decode_from_cell_chars()` created a second decode path that conflicted with stateful pattern decoding

## Fixes Applied

### Deterministic preprocessing

- `preprocess_frame()` now actually scores threshold variants and selects one
- preprocessing scoring is based on observed connected-component statistics rather than a fixed hardcoded choice
- perspective correction is now truly controlled by the caller

### Deterministic dot extraction

- replaced contour/debug-print heavy logic with connected-component dot detection
- added merged-component splitting for oversized blobs
- component sizing is inferred from the image itself instead of fixed global assumptions

### Deterministic segmentation

- line clustering, column clustering, cell grouping, and word-gap handling are now separate steps
- word-break insertion now uses cell-to-cell gaps instead of raw dot-column gaps
- 2x3 matrices are built explicitly and preserved in the response payload

### Deterministic pattern encoding

- row centers are estimated per line
- each cell now produces an explicit 2x3 occupancy matrix and one canonical 6-bit pattern
- one-column cells are resolved with left/right hypotheses against known Braille patterns

### Deterministic translation

- translation now runs through one stateful decode path
- per-cell display chars come from the same translator that builds output text

### Debugging improvements

- structured `RecognitionResult.debug` metadata now records:
  - selected preprocessing variant
  - candidate variant scores
  - line count
  - dot count
  - row centers
  - column centers
  - within-cell gap estimate
  - word-gap estimate
- annotated overlays now show:
  - detected dots
  - inferred row guides
  - cell boxes
  - per-cell pattern labels
  - confidence percentages
- app and CLI entry points now initialize standard logging so module logs are visible

## Verification

Regression command used:

```bash
python backend/scripts/eval_regression.py
```

Latest result after the refactor:

- exact matches: `8 / 9`
- mean character accuracy: `0.9861`

Exact matches now pass for:

- synthetic: `you can do it !`
- synthetic: `tiny cat`
- sample: `abc`
- sample: `hello`
- sample: `braille`
- sample: `i love you`
- sample: `good morning`

## Remaining Issues

One regression case still fails exactly:

- synthetic: `this is a great city`
  - output: `this is a gre?b city`

What this indicates:

- the pipeline is now stable on clean single-word and mixed-width sample inputs
- one edge case remains in column-to-cell grouping when a local region mixes different cell widths in a long strip
- this is a segmentation ambiguity, not a threshold failure

Recommended next work:

1. Add a second-pass local consistency check for suspicious cells with unusually wide bounding boxes.
2. Re-segment those cells by comparing neighboring cell pitch and expected two-column spacing.
3. Expand regression cases for long mixed-width phrases so segmentation heuristics are validated before deployment.

## Summary

The main problem was architectural, not numerical:

- preprocessing options existed but were not active
- perspective flags existed but were ignored
- dot extraction, segmentation, encoding, and translation were intertwined
- multiple generations of heuristics were left in one file

The current pipeline is now:

- deterministic
- traceable
- debuggable
- much more reliable on clean Braille images

The ML classifier remains in the repository as a training utility, but the main OCR path is now pattern-first and geometry-first.
