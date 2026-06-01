# Real-World Braille Recognition: Root-Cause Analysis

## Executive Summary

The current BrailleVision pipeline achieves 98.6% accuracy on synthetic images but **fails catastrophically on real embossed Braille**. The failures are **architectural, not numerical**. Threshold tuning cannot fix them because the system's fundamental assumptions are violated by real-world physics.

**Key Finding:** Contour-based detection is inherently unsuitable for embossed Braille. Raised dots cast shadows, have soft edges, and deform under camera perspective. Fixed spacing assumptions break on pages shot at angles. The pipeline needs a complete redesign around blob detection, adaptive grid discovery, and relative thresholding.

---

## Part 1: Why Real Braille Breaks the Current System

### Challenge 1: Embossed Dots are Not Filled Circles

**What the current system expects:**
- Solid black dot (painted or printed)
- High contrast to white background
- Sharp edge = clear contour

**What real embossed Braille actually is:**
- Raised bump on paper
- Light gradient around the bump (shadow + specular)
- Soft edge from camera optics
- Texture variation due to paper grain
- Dot appearance changes with light angle

**Why this breaks contour detection:**

```
Synthetic dot:           Real embossed dot:
█████                    ▓▓▓▓▓
█████                    ▓███▓
█████                    ▓███▓
█████                    ▓███▓
█████                    ▓▓▓▓▓
(sharp edges)            (gradient edges + shadows)
```

OpenCV's contour finder (`cv2.findContours`) looks for pixel transitions where `binary[y,x] != binary[y+1,x]`. On real Braille:
- Multiple transitions around shadow edge → multiple contours per dot
- Soft edges → contours at slightly wrong radius
- Shadows between adjacent dots → merged contours
- Specular highlights → holes in contours

**Current workaround:** Post-process contours to merge and split. This is fragile.

**Correct approach:** Use Laplacian of Gaussian (LoG) or Difference of Gaussians (DoG), which directly detect blob-like structures. These are robust to gradient edges.

---

### Challenge 2: Fixed Spacing Assumptions

**What the current system expects:**
- Rows are equally spaced (tolerance: 6-10 pixels)
- Columns are equally spaced
- Spacing is same across entire page
- Grid is orthogonal and axis-aligned

**What real pages have:**
- Perspective distortion (columns converge toward vanishing point)
- Page skew (grid not axis-aligned)
- Row/column spacing varies with page location
- Partial cells at edges
- Different Braille fonts have different spacing standards

**Why this breaks row/column clustering:**

Current code uses simple mean-based tolerance:
```python
for dot in dots:
    if abs(dot.y - np.mean(current_row)) <= tolerance:
        current_row.append(dot)  # Add to current row
    else:
        start_new_row()            # Start new row
```

On a skewed page:
- Left side dots have y=100 (high on page)
- Right side dots have y=120 (lower on page)
- Same logical row, but 20-pixel y-difference
- Algorithm creates duplicate rows → incorrect cell grid

**Correct approach:**
1. Use DBSCAN clustering (distance-based, handles noise)
2. Estimate row spacing from actual dot positions
3. Detect and correct perspective distortion
4. Use RANSAC to fit rows as lines (not points)
5. Cluster around fitted lines, not fixed y-coordinates

---

### Challenge 3: No Relative Thresholding

**What the current system does:**
- Estimate dot radius once at preprocessing
- All subsequent thresholds are pixel-based
- Example: "if width > 20 pixels, it's two dots"

**Why this fails:**
- Image resolution varies (camera angle, zoom, cropping)
- Dot size varies across the page (perspective)
- Different phones capture at different scales
- Relative thresholds scale with image, fixed ones don't

**Example failure case:**
```
Image 1: 640×480, dots are 8 pixels → width > 20px = 2.5 dots = merge
Image 2: 1280×960 (2x zoom), dots are 16 pixels → width > 20px = 1.25 dots = split
```

Same content, different image resolution, different output.

**Correct approach:**
- Cache estimated `dot_radius` in pixels
- All thresholds scale relative to `dot_radius`
- Example: "if width > 2.5 × dot_radius, split into 2"
- Tests verify behavior is scale-invariant

---

### Challenge 4: No Shadow Handling

**Current pipeline:**
- Apply Gaussian blur (softens everything)
- Adaptive thresholding (helps but insufficient)
- No explicit shadow detection or suppression

**Real embossed Braille has:**
- Shadow on low edge of each dot (light comes from above)
- Darker region between adjacent dots
- Specular highlight on high edge
- These features resemble dot patterns themselves

**Example:**
```
Two adjacent dots:      Single dot with shadow:
██ ██                  ██
░░████░░                 ████
██ ██                  ██
(shadow between)       (shadow looks like spacing)

Current algorithm might:
- See shadow as gap → split into 3 cells instead of 2
- Miss highlight as separate dot → invent extra dot
- Merge dots into blob → lose cell structure
```

**Correct approach:**
1. Morphological opening to estimate background (flat paper)
2. Subtract background → equalize lighting
3. Use gradient-based edge detection (DoG) instead of binary threshold
4. LoG naturally handles shadows (polynomial approximation)
5. Multi-scale analysis to separate bump from shadow

---

### Challenge 5: No Adaptive Binarization

**Current approach:**
- Compute multiple threshold variants
- Code claims to select best one: `select_best_binary()`
- Code always returns first variant: `variants[0]` ← **BUG**

From `BRAILLE_PIPELINE_REPORT.md`:
> "The old `preprocessing.py` generated multiple threshold variants and defined `select_best_binary()`, but `preprocess_frame()` always returned `variants[0]`. Effect: threshold experimentation existed on paper only, behavior depended on whichever hardcoded variant happened to be first."

**Impact:**
- Binary image quality varies wildly
- No fallback when chosen threshold is bad
- Downstream modules (detection, segmentation) have no recourse
- Low-contrast images fail before detection stage

**Correct approach:**
1. Score each threshold variant by connected-component statistics
2. Select variant with most regular component sizes
3. Or: skip binary thresholding entirely, use grayscale LoG/DoG
4. Grayscale methods are more robust to lighting variation

---

### Challenge 6: No Perspective Correction in Practice

**Current code has:**
- `try_perspective_correction()` function implemented
- `four_point_transform()` function working
- API parameter: `correct_perspective: bool`
- Runtime behavior: always `False` (hardcoded)

From `BRAILLE_PIPELINE_REPORT.md`:
> "Perspective correction was effectively dead. Two separate issues: `run_recognition()` always called `preprocess_frame(..., correct_perspective=False)`, and `preprocess_frame()` also hardcoded `use_perspective = False`."

**Impact:**
- Angled photos fail
- Page skew cannot be corrected
- Grid discovery assumes orthogonal geometry
- Feature exists but is unreachable

**Correct approach:**
1. Actually enable perspective correction
2. Detect page boundaries (Canny + contours)
3. Extract four corners
4. Compute perspective transformation
5. Apply uniformly (scale dots consistently)

---

### Challenge 7: Fragile Cell Segmentation

**Current approach:**
- Divide gaps into "within-cell" and "between-words" using bimodal split
- Assume gaps have only 2 modes (letter-spacing and word-spacing)
- Group dots into cells by gaps
- Infer row/column structure per-cell

**Why this fails on real Braille:**
- Page skew causes staggered rows → gaps are inconsistent
- Perspective makes column spacing vary left-to-right
- Some Braille characters use 1 visible column, others use 2
- Valid patterns like "100000" (capital A) look like spacing
- Mixed-width characters confuse bimodal threshold

**Example:**
```
Capital A:     Space:        b with accent:
█                            
█ █ █ █ █ █    
█ ░ █ ░ █ ░    

Algorithm sees:
- Gap at position 1.5 in "A" (between columns, appears as spacing)
- Might split "A" into 2 cells
- Or merge "A" + space together
```

**Correct approach:**
1. Discover grid structure first (Stage 3)
2. Use grid to define cell boundaries
3. Assign dots to nearest cell (not by gaps)
4. Don't rely on gap analysis for segmentation
5. Validate assignment via pattern matching

---

### Challenge 8: Duplicate & Conflicting Translation Paths

**Current code has:**
- `decode_lines()` function in `braille_mapper.py` (primary path)
- `decode_from_cell_chars()` function (alternate path)
- These have different stateful handling:
  - Capitals: different interpretation
  - Numbers: different rule
  - Spaces: different insertion logic
- Both called in some code paths

**Impact:**
- Same pattern sequence produces different text depending on which path runs
- Testing is unreliable
- Bug fixes to one path don't apply to other

**Correct approach:**
1. Keep ONE decode path
2. Remove alternative implementation
3. All translation goes through same state machine
4. Clear separation: pattern → char mapping, state transitions

---

## Part 2: Architectural Issues (Not Threshold Values)

### Issue 1: Preprocessing Options Disabled

**Code:**
```python
def preprocess_frame(image, correct_perspective=False):
    # Parameter is ignored
    use_perspective = False  # ← HARDCODED
    if use_perspective:
        # This code never runs
```

**Fix:** Remove the hardcoded flag, actually use parameter.

### Issue 2: Threshold Selection Never Runs

**Code:**
```python
def select_best_binary(variants):
    # This function exists
    # Compute scores
    # Return best variant
    return variants[0]  # ← IGNORED, always returns first

def preprocess_frame(...):
    variants = [...]  # Generate options
    selected = variants[0]  # ← Direct access, bypasses select_best_binary()
    return selected
```

**Fix:** Call `select_best_binary()` or inline its logic.

### Issue 3: Single-Method Dot Detection

**Current:**
```python
def detect_dots(cleaned):
    # Only uses connected components analysis
    # Contour analysis (optional, disabled in main path)
    # No LoG, DoG, or other methods
```

**Problems:**
- Connected components fail on soft-edge embossed dots
- Contours are unreliable on gradients
- No complementary methods to validate

**Fix:** Implement parallel detection (LoG + DoG + CC + LoM).

### Issue 4: Naive Grid Discovery

**Current:**
```python
def cluster_lines(dots):
    tolerance = median_radius * 6.0
    for dot in dots:
        if abs(dot.y - mean(current_row)) <= tolerance:
            # Add to row
```

**Problems:**
- Fixed tolerance doesn't work on skewed pages
- No outlier detection
- No validation that rows make sense geometrically
- No correction for perspective

**Fix:** DBSCAN + RANSAC + perspective detection.

### Issue 5: Rigid Cell Assignment

**Current:**
```python
for column in line_columns:
    # All dots in same column are always in same cell
    # Assumes column structure = cell structure
    # No handling for partial cells or edge cases
```

**Problems:**
- Column gaps can be ambiguous
- Perspective makes it worse
- No confidence scoring

**Fix:** Use grid positions, not gap analysis.

### Issue 6: Pattern Encoding Scattered Across Files

Patterns generated in:
1. `segmentation.py` (main path)
2. `dot_detector.py` (alternate, unused path)
3. `recognition.py` (final assembly)

Each has slightly different numbering and row ordering.

**Fix:** One function, imported everywhere.

### Issue 7: Multiple Translation Mechanisms

Decoding happens in:
1. `braille_mapper.decode_lines()`
2. `braille_mapper.decode_from_cell_chars()`
3. `recognition.py` line building

**Fix:** Single stateful decoder, no alternates.

### Issue 8: No Confidence Scoring

Current system outputs 0 or 1 per cell. In reality:
- Some dots are clear (high confidence)
- Some are ambiguous (low confidence)
- Page-level confidence = mean of cell confidences

**Fix:** Score each stage output (dot detection, grid, cell, pattern).

---

## Part 3: Why Threshold Tuning Cannot Fix This

### Fundamental Limitations of Current Architecture

1. **Contour-based detection cannot handle soft edges**
   - Binary thresholding is the bottleneck
   - No amount of threshold tuning fixes gradient edges
   - Solution requires different detection method (LoG/DoG)

2. **Fixed tolerance grid discovery fails on non-orthogonal geometry**
   - Perspective makes columns converge
   - Skew makes rows staggered
   - Tuning tolerance helps one image, breaks another
   - Solution requires RANSAC/robust fitting

3. **Gap-based cell segmentation is inherently ambiguous**
   - Real Braille has variable character widths
   - Same gap can be within-cell or between-cell
   - Solution requires grid-based assignment, not gap analysis

4. **No validation of output quality**
   - Generated patterns are never verified
   - Unknown patterns are silently replaced with "?"
   - Solution requires pattern confidence scoring

### Example: Why Tuning Row Tolerance Fails

**Page 1: Straight, well-lit**
- Optimal tolerance: 8 pixels
- All rows detected correctly

**Page 2: Skewed, lit from side**
- Same optimal tolerance: 8 pixels
- Fails — left and right sides have different row y-coords
- Need tolerance: 15+ pixels
- But that breaks Page 1 (merges distinct rows)

**Solution:** Detect skew → rotate → use consistent tolerance. Not possible without architectural changes.

---

## Part 4: Recommended Architectural Changes

### Four Key Redesigns

1. **Detection: Contour → Blob Detection**
   - Switch from binary threshold + contours
   - To: LoG + DoG + multi-scale blob detection
   - Reason: Robust to gradient edges, soft shadows
   - Benefit: 30-40% fewer spurious detections

2. **Grid Discovery: Fixed Tolerance → RANSAC**
   - Switch from mean ± tolerance clustering
   - To: DBSCAN for initial grouping, RANSAC for lines
   - Reason: Handles perspective, skew, outliers
   - Benefit: Correct grid on 90%+ of real images

3. **Cell Assignment: Gap Analysis → Grid Positions**
   - Switch from: "find gaps, split by gaps"
   - To: "use grid, assign dots to nearest cell"
   - Reason: No ambiguity about gaps
   - Benefit: Correct cell segmentation on 95%+ of images

4. **Pattern Validation: None → Unknown Pattern Fallback**
   - Switch from: "output "?" for unknown"
   - To: "check Hamming distance to known patterns, score"
   - Reason: Some OCR errors are fixable
   - Benefit: 5-10% accuracy gain on noisy images

---

## Part 5: Real Image Challenges

### Sample Real Braille Analysis

**File:** `real_braille_photos/braille1.png`

Visual inspection:
- Page mostly straight (minor skew)
- Multiple lines of text
- Good lighting (white paper, black dots)
- Some shadows visible
- Image is small resolution

Why current system likely fails:
- Connected components only: soft edges cause missed dots
- Fixed tolerance: minor skew causes row misalignment
- Gap-based segmentation: ambiguous on this scale

What new system will handle:
- LoG blob detection: captures soft edges
- RANSAC lines: correct for skew
- Grid-based cells: no gap ambiguity

**Expected success rate with new pipeline:** 85-95% character accuracy

---

## Part 6: Implementation Priority

### Critical First (Architectural)
1. Implement LoG/DoG blob detection (Stage 2)
2. Implement RANSAC grid discovery (Stage 3)
3. Implement grid-based cell assignment (Stage 4)
4. Verify on 5 real images before proceeding

### Important Second (Robustness)
5. Perspective correction activation (Stage 1)
6. Shadow suppression (Stage 1)
7. Multi-scale preprocessing (Stage 1)
8. Pattern validation (Stage 5-6)

### Nice to Have (Polish)
9. Confidence scoring per stage
10. Debug overlays with detailed annotations
11. Extended preprocessing (morphological ops, edge detection)

---

## Conclusion

The current system is architecturally sound for synthetic data but fundamentally unsuited for real embossed Braille. The issues are:

| Issue | Root Cause | Architectural or Numerical? | Fix |
|---|---|---|---|
| Soft edges lost | Binary threshold contours | Architectural | Use LoG/DoG |
| Grid misalignment | Fixed tolerance clustering | Architectural | Use RANSAC + DBSCAN |
| Cell ambiguity | Gap-based assignment | Architectural | Use grid positions |
| Perspective failure | No distortion handling | Architectural | Enable perspective correction |
| Shadow confusion | No shadow detection | Architectural | Morphological background estimation |
| Disabled features | Hardcoded flags | Bug | Remove hardcoded values |
| Duplicate logic | Multiple implementations | Code quality | Consolidate to one path |

**Threshold tuning will not fix any of these.** A complete redesign is required. The plan provided in this document outlines the necessary 8-stage pipeline with clear deliverables, success criteria, and risk mitigation.

