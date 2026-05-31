"""
Sparsh Dot Detector — dot finding, gap-classified line segmentation, centroid grid reading.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

from .ml_model import classify_cell


@dataclass
class BrailleCell:
    col: int
    row: int
    pattern: str
    char: str
    bbox: tuple
    dot_centers: list = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class DetectionResult:
    cells: list[BrailleCell]
    lines: list[list[BrailleCell]]
    patterns_by_line: list[list[str]]
    dot_radius_est: float
    cell_width_est: float
    cell_height_est: float
    annotated_image: np.ndarray


# ─── Dot detection ───────────────────────────────────────────────────────────

def count_dot_candidates(cleaned: np.ndarray) -> int:
    return len(_find_dots_on_binary(cleaned))


def _find_dots_on_binary(cleaned: np.ndarray) -> list[tuple[int, int, int]]:
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots: list[tuple[int, int, int]] = []
    h, w = cleaned.shape[:2]
    max_area = max(400, (h * w) // 35)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 6 or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.38:
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            r = max(2, int(np.sqrt(area / np.pi)))
            dots.append((cx, cy, r))

    if not dots:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 6
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = 0.38
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cleaned)
        dots = [(int(kp.pt[0]), int(kp.pt[1]), max(2, int(kp.size / 2))) for kp in keypoints]

    return dots


def _merge_dots(dots: list[tuple[int, int, int]], merge_dist: float) -> list[tuple[int, int, int]]:
    if not dots:
        return []
    merged: list[tuple[int, int, int]] = []
    used = [False] * len(dots)
    for i, (x1, y1, r1) in enumerate(dots):
        if used[i]:
            continue
        cluster = [(x1, y1, r1)]
        used[i] = True
        for j in range(i + 1, len(dots)):
            if used[j]:
                continue
            x2, y2, r2 = dots[j]
            if np.hypot(x1 - x2, y1 - y2) < merge_dist:
                cluster.append((x2, y2, r2))
                used[j] = True
        merged.append(
            (
                int(np.mean([d[0] for d in cluster])),
                int(np.mean([d[1] for d in cluster])),
                max(2, int(np.mean([d[2] for d in cluster]))),
            )
        )
    return merged


def detect_dots(cleaned: np.ndarray) -> list[tuple[int, int, int]]:
    dots = _find_dots_on_binary(cleaned)
    if len(dots) < 4:
        inv = cv2.bitwise_not(cleaned)
        dots.extend(_find_dots_on_binary(inv))
    return _merge_dots(dots, merge_dist=max(7.0, min(cleaned.shape) / 80.0))


def cluster_lines(dots: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    if not dots:
        return []

    sorted_dots = sorted(dots, key=lambda d: d[1])
    y_coords = np.array([d[1] for d in sorted_dots])

    if len(y_coords) > 1:
        diffs = np.diff(y_coords)
        non_zero = diffs[diffs > 2]
        avg_spacing = float(np.median(non_zero)) if len(non_zero) > 0 else 15.0
    else:
        avg_spacing = 15.0

    lines: list[list[tuple[int, int, int]]] = []
    current_line = [sorted_dots[0]]

    for dot in sorted_dots[1:]:
        last_y = float(np.mean([d[1] for d in current_line]))
        if dot[1] - last_y < avg_spacing * 3.2:
            current_line.append(dot)
        else:
            lines.append(current_line)
            current_line = [dot]
    if current_line:
        lines.append(current_line)

    return lines


# ─── Gap analysis: largest-jump split (within-cell vs between-letter vs word) ─

def _gap_thresholds(gaps: np.ndarray, dot_radius: float) -> tuple[float, float]:
    """
    letter_split: gaps above this start a new cell (between letters).
    word_split: gaps above this insert a space cell (between words).
    Uses percentiles so long sentences stay stable.
    """
    if len(gaps) == 0:
        base = dot_radius * 2.5
        return base * 1.7, base * 3.2

    gaps = np.sort(np.asarray(gaps, dtype=np.float64))
    positive = gaps[gaps > max(3.0, dot_radius * 0.35)]
    if len(positive) == 0:
        base = dot_radius * 2.5
        return base * 1.7, base * 3.2

    letter_split = float(np.percentile(positive, 48)) * 1.06
    letter_split = max(letter_split, dot_radius * 2.0)

    word_split = float(np.percentile(positive, 90)) * 0.92
    word_split = max(word_split, letter_split * 1.62)

    return letter_split, word_split


def _dedupe_dots_along_x(
    sorted_dots: list[tuple[int, int, int]], dot_radius: float
) -> list[tuple[int, int, int]]:
    """Merge only true duplicate blobs (same x AND same y within a tiny radius)."""
    if len(sorted_dots) <= 1:
        return sorted_dots
    # Only merge if BOTH x and y are nearly identical (true duplicates from double-detection)
    xy_sep = max(3.0, dot_radius * 0.4)
    out: list[tuple[int, int, int]] = [sorted_dots[0]]
    for dot in sorted_dots[1:]:
        prev = out[-1]
        if abs(dot[0] - prev[0]) < xy_sep and abs(dot[1] - prev[1]) < xy_sep:
            # True duplicate — merge
            out[-1] = (
                int((prev[0] + dot[0]) / 2),
                int((prev[1] + dot[1]) / 2),
                max(prev[2], dot[2]),
            )
        else:
            out.append(dot)
    return out


def _find_bimodal_split(gaps: np.ndarray, dot_radius: float) -> tuple[float, float]:
    """
    Find the letter_split threshold by detecting the bimodal gap distribution.
    Intra-cell gaps (between dots in the same cell) are small.
    Inter-cell gaps (between cells) are larger.
    Word gaps are the largest.

    Returns (letter_split, word_split).
    """
    positive = gaps[gaps > max(2.0, dot_radius * 0.3)]
    if len(positive) == 0:
        base = dot_radius * 2.5
        return base * 1.5, base * 3.0

    sorted_gaps = np.sort(positive)

    # Look for the largest relative jump in sorted gaps — that's the intra/inter boundary
    if len(sorted_gaps) >= 2:
        ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1.0)
        # Find the biggest jump — this separates intra-cell from inter-cell gaps
        jump_idx = int(np.argmax(ratios))
        letter_split = float(sorted_gaps[jump_idx]) * 1.15
        letter_split = max(letter_split, dot_radius * 1.8)
    else:
        letter_split = float(sorted_gaps[0]) * 1.5
        letter_split = max(letter_split, dot_radius * 1.8)

    # Word split: only insert spaces for gaps that are clearly larger than ALL inter-cell gaps.
    # Use a robust bimodal detection: check if inter-cell gaps form 2 distinct clusters.
    inter_cell_gaps = positive[positive > letter_split]
    if len(inter_cell_gaps) >= 3:
        sorted_inter = np.sort(inter_cell_gaps)
        # Find the largest relative jump within inter-cell gaps
        inter_ratios = sorted_inter[1:] / np.maximum(sorted_inter[:-1], 1.0)
        max_ratio = float(np.max(inter_ratios))
        jump_idx = int(np.argmax(inter_ratios))

        # Only treat as word gap if:
        # 1. The ratio jump is significant (>= 1.35)
        # 2. There are at least 2 gaps on the larger side (multiple word gaps)
        # 3. The word gap is at least 3x the letter_split (absolute minimum)
        n_large = len(sorted_inter) - jump_idx - 1
        candidate_word_gap = float(sorted_inter[jump_idx + 1]) if jump_idx + 1 < len(sorted_inter) else 0.0
        if max_ratio >= 1.35 and n_large >= 2 and candidate_word_gap >= letter_split * 4.0:
            word_split = float(sorted_inter[jump_idx]) * 1.15
            word_split = max(word_split, letter_split * 2.5)
        else:
            # No clear word gap — suppress spaces
            word_split = float(np.max(positive)) * 3.0
    elif len(inter_cell_gaps) == 2:
        sorted_inter = np.sort(inter_cell_gaps)
        ratio = sorted_inter[1] / max(sorted_inter[0], 1.0)
        if ratio >= 1.8 and sorted_inter[1] >= letter_split * 3.0:
            word_split = float(sorted_inter[0]) * 1.3
        else:
            word_split = float(np.max(positive)) * 3.0
    elif len(inter_cell_gaps) == 1:
        if inter_cell_gaps[0] > letter_split * 3.5:
            word_split = float(inter_cell_gaps[0]) * 0.9
        else:
            word_split = float(inter_cell_gaps[0]) * 3.0
    else:
        word_split = letter_split * 4.0

    return letter_split, word_split


def _group_dots_into_cells(
    sorted_dots: list[tuple[int, int, int]],
    dot_radius: float,
) -> list[list[tuple[int, int, int]]]:
    """Segment dots into Braille cells using bimodal gap analysis."""
    if not sorted_dots:
        return []
    if len(sorted_dots) == 1:
        return [sorted_dots]

    sorted_dots = _dedupe_dots_along_x(sorted_dots, dot_radius)

    xs = np.array([d[0] for d in sorted_dots], dtype=np.float64)
    gaps = np.diff(xs)

    letter_split, word_split = _find_bimodal_split(gaps, dot_radius)

    allow_spaces = len(sorted_dots) > 8

    groups: list[list[tuple[int, int, int]]] = []
    current = [sorted_dots[0]]

    for i, gap in enumerate(gaps):
        nxt = sorted_dots[i + 1]
        if gap <= letter_split:
            current.append(nxt)
        else:
            groups.append(current)
            if allow_spaces and gap >= word_split:
                groups.append([])
            current = [nxt]
    groups.append(current)

    return groups


# ─── Pattern from dots ───────────────────────────────────────────────────────

def _pattern_flat_row(
    group: list[tuple[int, int, int]], dot_radius: float
) -> tuple[str, list[float], list[tuple[int, int]]]:
    """Flat cells (all dots at same y): assign by column position only."""
    # Sort by x
    sorted_group = sorted(group, key=lambda d: d[0])
    x_mid = float(np.median([d[0] for d in sorted_group]))

    left_dots = [d for d in sorted_group if d[0] <= x_mid]
    right_dots = [d for d in sorted_group if d[0] > x_mid]

    pattern = ["0"] * 6
    intensities = [0.0] * 6
    matched: list[tuple[int, int]] = []

    for i, dot in enumerate(left_dots[:3]):
        pattern[i] = "1"
        intensities[i] = 0.88
        matched.append((dot[0], dot[1]))

    for i, dot in enumerate(right_dots[:3]):
        pattern[3 + i] = "1"
        intensities[3 + i] = 0.88
        matched.append((dot[0], dot[1]))

    return "".join(pattern), intensities, matched


def _pattern_full_grid(
    group: list[tuple[int, int, int]], dot_radius: float
) -> tuple[str, list[float], list[tuple[int, int]]]:
    """
    Assign dots to Braille slots 0-5 (dot1..dot6) using column/row classification.

    Braille cell layout:
      slot 0 (dot1) = left col, top row
      slot 1 (dot2) = left col, mid row
      slot 2 (dot3) = left col, bot row
      slot 3 (dot4) = right col, top row
      slot 4 (dot5) = right col, mid row
      slot 5 (dot6) = right col, bot row

    Strategy:
    1. Split dots into left/right columns by x-median.
    2. Estimate the row pitch from all dots in the group (or from dot_radius).
    3. Assign each dot to top/mid/bot based on its y relative to the cell's
       estimated top y and row pitch.
    """
    xs = [d[0] for d in group]
    ys = [d[1] for d in group]

    x_mid = float(np.median(xs))
    left_dots = [d for d in group if d[0] <= x_mid]
    right_dots = [d for d in group if d[0] > x_mid]

    # If all dots are on one side, use mean instead of median
    if not left_dots or not right_dots:
        x_mean = float(np.mean(xs))
        left_dots = [d for d in group if d[0] <= x_mean]
        right_dots = [d for d in group if d[0] > x_mean]

    pattern = ["0"] * 6
    intensities = [0.0] * 6
    matched: list[tuple[int, int]] = []

    # Estimate row pitch from all dots in the group
    # In a Braille cell, rows are equally spaced. Estimate pitch from y-spread.
    y_vals_sorted = sorted(set(d[1] for d in group))
    if len(y_vals_sorted) >= 3:
        # 3 distinct y-levels → pitch = (max_y - min_y) / 2
        row_pitch = (y_vals_sorted[-1] - y_vals_sorted[0]) / 2.0
    elif len(y_vals_sorted) == 2:
        # 2 distinct y-levels: could be rows 0+1, 0+2, or 1+2
        # We can't know which without more context, but we can estimate:
        # If the gap is close to dot_radius*2, it's adjacent rows (pitch = gap)
        # If the gap is larger, it's rows 0+2 (pitch = gap/2)
        gap = float(y_vals_sorted[1] - y_vals_sorted[0])
        # Use dot_radius to estimate: adjacent rows are ~2*dot_radius apart
        if gap < dot_radius * 3.5:
            row_pitch = gap  # adjacent rows
        else:
            row_pitch = gap / 2.0  # skipped middle row
    else:
        row_pitch = dot_radius * 2.2

    row_pitch = max(row_pitch, dot_radius * 1.2)

    # Estimate cell top y
    y_top = float(min(ys))

    def y_to_row(y: float) -> int:
        """Map a y coordinate to row index 0 (top), 1 (mid), 2 (bot)."""
        rel = (y - y_top) / row_pitch  # 0=top, 1=mid, 2=bot
        if rel < 0.5:
            return 0
        elif rel < 1.5:
            return 1
        else:
            return 2

    def assign_col(col_dots: list[tuple[int, int, int]], col_offset: int) -> None:
        for dot in col_dots[:3]:
            row = y_to_row(float(dot[1]))
            slot = col_offset + row
            if pattern[slot] == "0":
                pattern[slot] = "1"
                intensities[slot] = 0.88
                matched.append((dot[0], dot[1]))

    assign_col(left_dots, 0)   # slots 0,1,2
    assign_col(right_dots, 3)  # slots 3,4,5

    return "".join(pattern), intensities, matched


def _canonical_grid_pattern(
    group: list[tuple[int, int, int]], dot_radius: float
) -> tuple[str, list[float], list[tuple[int, int]]]:
    if not group:
        return "000000", [0.0] * 6, []
    ys = [d[1] for d in group]
    if max(ys) - min(ys) < dot_radius * 2.2:
        return _pattern_flat_row(group, dot_radius)
    return _pattern_full_grid(group, dot_radius)


def fit_grid_to_group(group: list[tuple[int, int, int]], col_idx: int, dot_radius: float) -> BrailleCell:
    if not group:
        return BrailleCell(
            col=col_idx,
            row=0,
            pattern="000000",
            char=" ",
            bbox=(0, 0, 0, 0),
            dot_centers=[],
            confidence=0.85,
        )

    xs = [d[0] for d in group]
    ys = [d[1] for d in group]
    pad = int(dot_radius * 0.9)
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    binary_pattern, intensities, matched_dots = _canonical_grid_pattern(group, dot_radius)
    pattern, display_char, confidence = classify_cell(binary_pattern, intensities)

    if pattern == "000000" and display_char == "·":
        display_char = "?"

    return BrailleCell(
        col=col_idx,
        row=0,
        pattern=pattern,
        char=display_char if display_char != "·" else "?",
        bbox=bbox,
        dot_centers=matched_dots,
        intensities=intensities,
        confidence=confidence,
    )


def segment_cells_in_line(line_dots: list[tuple[int, int, int]], dot_radius: float) -> list[BrailleCell]:
    if not line_dots:
        return []

    sorted_dots = sorted(line_dots, key=lambda d: d[0])
    groups = _group_dots_into_cells(sorted_dots, dot_radius)

    cells: list[BrailleCell] = []
    for col_idx, group in enumerate(groups):
        cells.append(fit_grid_to_group(group, col_idx, dot_radius))

    return cells


def detect_braille(cleaned: np.ndarray, original: np.ndarray) -> DetectionResult:
    dots = detect_dots(cleaned)

    if not dots:
        return DetectionResult(
            cells=[],
            lines=[],
            patterns_by_line=[],
            dot_radius_est=5.0,
            cell_width_est=20.0,
            cell_height_est=30.0,
            annotated_image=original.copy(),
        )

    median_r = float(np.median([d[2] for d in dots]))
    line_groups = cluster_lines(dots)

    all_cells: list[BrailleCell] = []
    lines: list[list[BrailleCell]] = []
    patterns_by_line: list[list[str]] = []

    for line_idx, line_dots in enumerate(line_groups):
        line_cells = segment_cells_in_line(line_dots, median_r)
        for c in line_cells:
            c.row = line_idx
        all_cells.extend(line_cells)
        lines.append(line_cells)
        patterns_by_line.append([c.pattern for c in line_cells])

    annotated = annotate_hud(original, dots, all_cells)

    return DetectionResult(
        cells=all_cells,
        lines=lines,
        patterns_by_line=patterns_by_line,
        dot_radius_est=median_r,
        cell_width_est=median_r * 4.0,
        cell_height_est=median_r * 6.0,
        annotated_image=annotated,
    )


_LABEL_COLORS = [
    (180, 80, 255),
    (80, 80, 255),
    (80, 255, 120),
    (80, 200, 255),
    (255, 180, 80),
    (255, 120, 180),
    (120, 255, 220),
    (220, 255, 120),
]


def annotate_hud(
    image: np.ndarray,
    dots: list[tuple[int, int, int]],
    cells: list[BrailleCell],
) -> np.ndarray:
    hud = image.copy()

    for cx, cy, r in dots:
        cv2.circle(hud, (cx, cy), int(r * 1.5), (0, 230, 255), 1, cv2.LINE_AA)
        cv2.circle(hud, (cx, cy), max(2, int(r * 0.55)), (0, 255, 120), -1, cv2.LINE_AA)

    for i, cell in enumerate(cells):
        x, y, w, h = cell.bbox
        if w <= 0 or h <= 0:
            continue

        color = _LABEL_COLORS[i % len(_LABEL_COLORS)]
        cv2.rectangle(hud, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        label = "space" if cell.pattern == "000000" and cell.char == " " else f"{cell.char} {int(cell.confidence * 100)}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(0.55, w / 80.0))
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        tx, ty = max(0, x), max(th + 4, y - 6)
        cv2.rectangle(hud, (tx, ty - th - 4), (tx + tw + 4, ty + 2), (20, 20, 20), -1)
        cv2.putText(hud, label, (tx + 2, ty), font, scale, color, 1, cv2.LINE_AA)

    return hud
