"""
Braille line segmentation, 2x3 matrix generation, and pattern encoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .braille_mapper import get_all_patterns, pattern_to_char
from .dot_detector import render_dot_overlay
from .pipeline_types import BrailleCell, BrailleLine, DetectionResult, Dot
from .validation import log_segmentation_diagnostics, validate_cell_pattern, validate_cell_geometry


LOGGER = logging.getLogger(__name__)
KNOWN_PATTERNS = set(get_all_patterns().keys())
KNOWN_PATTERNS.update({"000001", "010011", "000000"})


@dataclass
class ColumnCluster:
    dots: list[Dot]
    center_x: float


def _cluster_scalar(values: list[float], tolerance: float) -> list[list[float]]:
    if not values:
        return []
    sorted_values = sorted(values)
    groups: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - np.mean(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def cluster_lines(dots: list[Dot]) -> list[list[Dot]]:
    if not dots:
        return []
    median_radius = float(np.median([dot.radius for dot in dots]))
    tolerance = max(10.0, median_radius * 6.0)
    lines: list[list[Dot]] = [[min(dots, key=lambda dot: dot.y)]]
    for dot in sorted(dots, key=lambda item: (item.y, item.x))[1:]:
        if abs(dot.y - np.mean([item.y for item in lines[-1]])) <= tolerance:
            lines[-1].append(dot)
        else:
            lines.append([dot])
    return [sorted(line, key=lambda dot: dot.x) for line in lines]


def _cluster_columns(line_dots: list[Dot]) -> list[ColumnCluster]:
    if not line_dots:
        return []
    median_radius = float(np.median([dot.radius for dot in line_dots]))
    tolerance = max(6.0, median_radius * 1.6)
    columns: list[list[Dot]] = [[line_dots[0]]]
    for dot in sorted(line_dots, key=lambda item: item.x)[1:]:
        current_center = float(np.mean([item.x for item in columns[-1]]))
        if abs(dot.x - current_center) <= tolerance:
            columns[-1].append(dot)
        else:
            columns.append([dot])
    return [
        ColumnCluster(dots=column, center_x=float(np.mean([dot.x for dot in column])))
        for column in columns
    ]


def _bimodal_threshold(gaps: list[float], fallback: float) -> float:
    positive = sorted(gap for gap in gaps if gap > 0)
    if len(positive) < 2:
        return fallback

    best_threshold = fallback
    best_cost = float("inf")
    for split in range(1, len(positive)):
        low = np.array(positive[:split], dtype=np.float32)
        high = np.array(positive[split:], dtype=np.float32)
        if len(low) == 0 or len(high) == 0:
            continue
        if np.mean(high) < np.mean(low) * 1.25:
            continue
        cost = float(np.var(low) * len(low) + np.var(high) * len(high))
        if cost < best_cost:
            best_cost = cost
            best_threshold = float((low[-1] + high[0]) / 2.0)
    return best_threshold


def _word_threshold(gaps: list[float]) -> float:
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return float("inf")
    baseline = float(np.percentile(np.array(positive, dtype=np.float32), 35))
    threshold = baseline * 1.75
    if max(positive) >= threshold:
        return threshold
    return float("inf")


def _pair_columns_into_cells(columns: list[ColumnCluster]) -> tuple[list[list[ColumnCluster]], float]:
    if not columns:
        return [], 0.0

    gaps = [columns[idx + 1].center_x - columns[idx].center_x for idx in range(len(columns) - 1)]
    median_radius = float(np.median([dot.radius for column in columns for dot in column.dots]))
    within_threshold = _bimodal_threshold(gaps, fallback=median_radius * 2.6)

    groups: list[list[ColumnCluster]] = []
    idx = 0
    while idx < len(columns):
        if idx + 1 < len(columns) and (columns[idx + 1].center_x - columns[idx].center_x) <= within_threshold:
            groups.append([columns[idx], columns[idx + 1]])
            idx += 2
        else:
            groups.append([columns[idx]])
            idx += 1
    return groups, within_threshold


def _group_gap(left: list[ColumnCluster], right: list[ColumnCluster]) -> float:
    return right[0].center_x - left[-1].center_x


def _row_center_candidates(line_dots: list[Dot]) -> list[tuple[float, float, float]]:
    median_radius = float(np.median([dot.radius for dot in line_dots]))
    tolerance = max(4.0, median_radius * 0.95)
    bands = _cluster_scalar([float(dot.y) for dot in line_dots], tolerance)
    centers = [float(np.mean(band)) for band in bands]

    if len(centers) >= 3:
        return [(centers[0], centers[len(centers) // 2], centers[-1])]
    if len(centers) == 2:
        first, second = centers
        gap = second - first
        if gap > median_radius * 3.2:
            return [(first, first + gap / 2.0, second)]
        return [
            (first, second, second + gap),
            (first - gap, first, second),
        ]
    if len(centers) == 1:
        first = centers[0]
        pitch = max(8.0, median_radius * 2.6)
        return [
            (first, first + pitch, first + pitch * 2.0),
            (first - pitch, first, first + pitch),
            (first - pitch * 2.0, first - pitch, first),
        ]
    return [(0.0, median_radius * 2.6, median_radius * 5.2)]


def _pattern_from_matrix(matrix: list[list[int]]) -> str:
    return "".join(str(matrix[col][row]) for col in range(2) for row in range(3))


def _cell_bbox(columns: list[ColumnCluster], pad: float) -> tuple[int, int, int, int]:
    xs = [dot.x for column in columns for dot in column.dots]
    ys = [dot.y for column in columns for dot in column.dots]
    x0 = int(round(min(xs) - pad))
    y0 = int(round(min(ys) - pad))
    x1 = int(round(max(xs) + pad))
    y1 = int(round(max(ys) + pad))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _encode_columns(
    columns: list[ColumnCluster],
    row_centers: tuple[float, float, float],
) -> tuple[str, list[list[int]], list[float], list[tuple[int, int]], float]:
    radius = float(np.median([dot.radius for column in columns for dot in column.dots]))
    dot_centers = [(dot.x, dot.y) for column in columns for dot in column.dots]

    def encode(slot_columns: list[int]) -> tuple[str, list[list[int]], list[float], float]:
        matrix = [[0, 0, 0], [0, 0, 0]]
        intensities = [0.0] * 6
        row_errors: list[float] = []
        col_errors: list[float] = []
        for column_idx, column in enumerate(columns):
            x_center = float(np.mean([dot.x for dot in column.dots]))
            for dot in column.dots:
                row_idx = int(np.argmin([abs(dot.y - row_center) for row_center in row_centers]))
                slot_col = slot_columns[column_idx]
                matrix[slot_col][row_idx] = 1
                intensities[slot_col * 3 + row_idx] = 1.0
                row_pitch = max(6.0, np.mean(np.diff(row_centers)) if len(row_centers) > 1 else radius * 2.6)
                row_errors.append(abs(dot.y - row_centers[row_idx]) / row_pitch)
                col_errors.append(abs(dot.x - x_center) / max(radius * 2.0, 6.0))
        pattern = _pattern_from_matrix(matrix)
        position_error = float(np.mean(row_errors + col_errors)) if (row_errors or col_errors) else 0.0
        confidence = max(0.2, min(0.99, 1.0 - position_error * 0.35))
        if pattern in KNOWN_PATTERNS:
            confidence = min(0.99, confidence + 0.08)
        return pattern, matrix, intensities, confidence

    if len(columns) == 1:
        left_pattern, left_matrix, left_intensities, left_conf = encode([0])
        right_pattern, right_matrix, right_intensities, right_conf = encode([1])
        left_known = left_pattern in KNOWN_PATTERNS
        right_known = right_pattern in KNOWN_PATTERNS
        if left_known and not right_known:
            return left_pattern, left_matrix, left_intensities, dot_centers, left_conf
        if right_known and not left_known:
            return right_pattern, right_matrix, right_intensities, dot_centers, right_conf
        return left_pattern, left_matrix, left_intensities, dot_centers, left_conf

    pattern, matrix, intensities, confidence = encode([0, 1])
    return pattern, matrix, intensities, dot_centers, confidence


def _build_cell(
    row_idx: int,
    col_idx: int,
    columns: list[ColumnCluster],
    row_centers: tuple[float, float, float],
) -> BrailleCell:
    pattern, matrix, intensities, dot_centers, confidence = _encode_columns(columns, row_centers)
    char = pattern_to_char(pattern)
    if char.startswith("__"):
        display_char = char.replace("__", "")
    elif char == "?":
        display_char = "?"
    else:
        display_char = char
    radius = float(np.median([dot.radius for column in columns for dot in column.dots]))
    return BrailleCell(
        col=col_idx,
        row=row_idx,
        pattern=pattern,
        char=display_char,
        bbox=_cell_bbox(columns, pad=radius),
        dot_centers=dot_centers,
        intensities=intensities,
        confidence=round(confidence, 3),
        matrix=matrix,
    )


def _blank_cell(row_idx: int, col_idx: int, anchor_x: float, anchor_y: float, radius: float) -> BrailleCell:
    width = int(round(max(10.0, radius * 3.0)))
    height = int(round(max(16.0, radius * 4.5)))
    return BrailleCell(
        col=col_idx,
        row=row_idx,
        pattern="000000",
        char=" ",
        bbox=(int(round(anchor_x)), int(round(anchor_y - height / 2.0)), width, height),
        confidence=1.0,
        matrix=[[0, 0, 0], [0, 0, 0]],
    )


def _score_row_candidate(
    candidate: tuple[float, float, float],
    groups: list[list[ColumnCluster]],
) -> float:
    score = 0.0
    for columns in groups:
        pattern, _, _, _, confidence = _encode_columns(columns, candidate)
        score += confidence
        if pattern in KNOWN_PATTERNS:
            score += 2.0
    return score


def _select_row_centers(line_dots: list[Dot], groups: list[list[ColumnCluster]]) -> tuple[float, float, float]:
    candidates = _row_center_candidates(line_dots)
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda candidate: _score_row_candidate(candidate, groups))


def annotate_segmentation(
    image: np.ndarray,
    dots: list[Dot],
    line_models: list[BrailleLine],
) -> np.ndarray:
    overlay = render_dot_overlay(image, dots)
    colors = [
        (255, 120, 80),
        (80, 220, 255),
        (120, 255, 120),
        (255, 200, 80),
    ]

    for line in line_models:
        line_color = colors[line.row % len(colors)]
        line_xs = [dot.x for dot in line.dots]
        if line_xs:
            x0 = int(min(line_xs) - 20)
            x1 = int(max(line_xs) + 20)
            for row_center in line.row_centers:
                cv2.line(
                    overlay,
                    (x0, int(round(row_center))),
                    (x1, int(round(row_center))),
                    line_color,
                    1,
                    cv2.LINE_AA,
                )

        for cell in line.cells:
            x, y, w, h = cell.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), line_color, 2, cv2.LINE_AA)
            if cell.matrix:
                col_centers = []
                for col_idx, column in enumerate(cell.matrix):
                    if any(column):
                        present = [point for point in cell.dot_centers if point[0] >= x and point[0] <= x + w]
                        if present:
                            local_points = [point[0] for point in present]
                            col_centers.append(int(round(np.mean(local_points))))
                for center_x in col_centers[:2]:
                    cv2.line(overlay, (center_x, y), (center_x, y + h), (255, 255, 255), 1, cv2.LINE_AA)

            label = f"{cell.char or '?'} {cell.pattern} {int(cell.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42
            (text_w, text_h), _ = cv2.getTextSize(label, font, scale, 1)
            text_x = max(0, x)
            text_y = max(text_h + 4, y - 6)
            cv2.rectangle(overlay, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y + 2), (16, 16, 16), -1)
            cv2.putText(overlay, label, (text_x + 2, text_y), font, scale, line_color, 1, cv2.LINE_AA)

    return overlay


def segment_braille_dots(dots: list[Dot], original: np.ndarray) -> DetectionResult:
    if not dots:
        return DetectionResult(
            cells=[],
            lines=[],
            patterns_by_line=[],
            dot_radius_est=0.0,
            cell_width_est=0.0,
            cell_height_est=0.0,
            annotated_image=original.copy(),
            debug={"line_count": 0, "dot_count": 0},
        )

    median_radius = float(np.median([dot.radius for dot in dots]))
    raw_lines = cluster_lines(dots)
    line_models: list[BrailleLine] = []
    all_cells: list[BrailleCell] = []
    lines: list[list[BrailleCell]] = []
    patterns_by_line: list[list[str]] = []
    max_cell_width = 0.0
    max_cell_height = 0.0

    for row_idx, line_dots in enumerate(raw_lines):
        columns = _cluster_columns(line_dots)
        groups, within_gap = _pair_columns_into_cells(columns)
        row_centers = _select_row_centers(line_dots, groups)
        cell_gaps = [_group_gap(groups[idx], groups[idx + 1]) for idx in range(len(groups) - 1)]
        word_gap = _word_threshold(cell_gaps)

        cells: list[BrailleCell] = []
        for group_idx, column_group in enumerate(groups):
            cell = _build_cell(row_idx, len(cells), column_group, row_centers)
            
            # Validate cell pattern
            validation = validate_cell_pattern(
                cell.pattern,
                len(cell.dot_centers),
                cell.dot_centers,
            )
            
            # Adjust confidence if warnings detected
            if validation.warnings:
                for warning in validation.warnings:
                    LOGGER.debug("Cell validation warning: %s", warning)
                cell.confidence = max(0.1, cell.confidence + validation.confidence_adjustment)
            
            # Log errors but don't reject cell
            if validation.issues:
                for issue in validation.issues:
                    LOGGER.warning("Cell validation error: %s", issue)
            
            cells.append(cell)

            if word_gap != float("inf") and group_idx < len(groups) - 1 and cell_gaps[group_idx] >= word_gap:
                anchor_x = column_group[-1].center_x + median_radius * 1.5
                anchor_y = float(np.mean(row_centers))
                cells.append(_blank_cell(row_idx, len(cells), anchor_x, anchor_y, median_radius))

        for col_idx, cell in enumerate(cells):
            cell.col = col_idx

        line_models.append(
            BrailleLine(
                row=row_idx,
                dots=line_dots,
                cells=cells,
                row_centers=row_centers,
                column_centers=[column.center_x for column in columns],
                within_cell_gap=within_gap,
                word_gap=word_gap,
            )
        )
        all_cells.extend(cells)
        lines.append(cells)
        patterns_by_line.append([cell.pattern for cell in cells])

        for cell in cells:
            max_cell_width = max(max_cell_width, float(cell.bbox[2]))
            max_cell_height = max(max_cell_height, float(cell.bbox[3]))

        # Enhanced logging
        log_segmentation_diagnostics(
            row_idx,
            line_dots,
            len(columns),
            cells,
            row_centers,
            within_gap,
            word_gap,
        )

    annotated = annotate_segmentation(original, dots, line_models)
    debug = {
        "line_count": len(line_models),
        "dot_count": len(dots),
        "lines": [
            {
                "row": line.row,
                "dot_count": len(line.dots),
                "cell_count": len(line.cells),
                "row_centers": [round(value, 2) for value in line.row_centers],
                "column_centers": [round(value, 2) for value in line.column_centers],
                "within_cell_gap": round(line.within_cell_gap, 2),
                "word_gap": None if line.word_gap == float("inf") else round(line.word_gap, 2),
            }
            for line in line_models
        ],
    }

    return DetectionResult(
        cells=all_cells,
        lines=lines,
        patterns_by_line=patterns_by_line,
        dot_radius_est=median_radius,
        cell_width_est=max_cell_width or (median_radius * 4.0),
        cell_height_est=max_cell_height or (median_radius * 6.0),
        annotated_image=annotated,
        debug=debug,
    )
