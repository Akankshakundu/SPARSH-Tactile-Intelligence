"""
Dot extraction for Braille OCR.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .pipeline_types import Dot


LOGGER = logging.getLogger(__name__)


def _component_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int, float]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])
        boxes.append((x, y, w, h, area))
    return boxes


def count_dot_candidates(cleaned: np.ndarray) -> int:
    return len(detect_dots(cleaned))


def _estimate_reference_area(binary: np.ndarray) -> float:
    image_area = float(binary.shape[0] * binary.shape[1])
    min_area = max(8.0, image_area * 0.000015)
    plausible = [area for _, _, _, _, area in _component_boxes(binary) if area >= min_area]
    if not plausible:
        return max(16.0, image_area * 0.00008)
    return float(np.median(np.array(plausible, dtype=np.float32)))


def _mask_from_box(binary: np.ndarray, box: tuple[int, int, int, int, float]) -> np.ndarray:
    x, y, w, h, _ = box
    return binary[y : y + h, x : x + w]


def _centroid_from_mask(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return float(np.mean(xs)), float(np.mean(ys))


def _split_large_component(
    binary: np.ndarray,
    box: tuple[int, int, int, int, float],
    reference_area: float,
) -> list[Dot]:
    x, y, w, h, area = box
    local = _mask_from_box(binary, box)
    expected_count = int(round(area / max(reference_area, 1.0)))
    expected_count = max(2, min(expected_count, 4))

    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    peak_level = max(1.0, np.max(dist) * 0.55)
    _, peaks = cv2.threshold(dist, peak_level, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    peak_count, _, _, peak_centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)

    centers: list[tuple[float, float]] = []
    for label in range(1, peak_count):
        cx, cy = peak_centroids[label]
        centers.append((float(cx), float(cy)))

    if len(centers) < 2:
        ys, xs = np.nonzero(local)
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        if len(points) < expected_count:
            return []
        _, _, centers = cv2.kmeans(
            points,
            expected_count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        centers = [tuple(center) for center in centers.tolist()]

    radius = max(2.0, np.sqrt(reference_area / np.pi))
    dots: list[Dot] = []
    for local_x, local_y in centers:
        dots.append(
            Dot(
                x=int(round(x + local_x)),
                y=int(round(y + local_y)),
                radius=radius,
                area=reference_area,
                bbox=(x, y, w, h),
            )
        )

    LOGGER.debug("Split merged component at (%d,%d,%d,%d) into %d dots", x, y, w, h, len(dots))
    return dots


def _deduplicate_dots(dots: list[Dot]) -> list[Dot]:
    if not dots:
        return []

    deduped: list[Dot] = []
    for dot in sorted(dots, key=lambda item: (item.x, item.y)):
        merged = False
        for idx, existing in enumerate(deduped):
            limit = max(existing.radius, dot.radius) * 0.55
            if np.hypot(existing.x - dot.x, existing.y - dot.y) <= limit:
                deduped[idx] = Dot(
                    x=int(round((existing.x + dot.x) / 2)),
                    y=int(round((existing.y + dot.y) / 2)),
                    radius=max(existing.radius, dot.radius),
                    area=max(existing.area, dot.area),
                    bbox=existing.bbox,
                )
                merged = True
                break
        if not merged:
            deduped.append(dot)
    return deduped


def detect_dots(cleaned: np.ndarray) -> list[Dot]:
    binary = cleaned if cleaned.ndim == 2 else cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    reference_area = _estimate_reference_area(binary)
    image_area = float(binary.shape[0] * binary.shape[1])
    min_area = max(8.0, image_area * 0.000015)
    max_single_area = max(reference_area * 2.8, 120.0)

    detected: list[Dot] = []
    for box in _component_boxes(binary):
        x, y, w, h, area = box
        if area < min_area:
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        if area <= max_single_area and aspect <= 2.4:
            mask = _mask_from_box(binary, box)
            local_x, local_y = _centroid_from_mask(mask)
            radius = max(2.0, np.sqrt(area / np.pi))
            detected.append(
                Dot(
                    x=int(round(x + local_x)),
                    y=int(round(y + local_y)),
                    radius=radius,
                    area=area,
                    bbox=(x, y, w, h),
                )
            )
            continue

        detected.extend(_split_large_component(binary, box, reference_area))

    deduped = _deduplicate_dots(detected)
    LOGGER.info("Detected %d Braille dots", len(deduped))
    return deduped


def render_dot_overlay(image: np.ndarray, dots: list[Dot]) -> np.ndarray:
    overlay = image.copy()
    for dot in dots:
        cv2.circle(overlay, (dot.x, dot.y), int(round(dot.radius * 1.5)), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (dot.x, dot.y), max(2, int(round(dot.radius * 0.45))), (80, 255, 120), -1, cv2.LINE_AA)
    return overlay
