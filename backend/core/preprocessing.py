"""
Image preprocessing for the deterministic Braille OCR pipeline.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class BinaryVariant:
    name: str
    image: np.ndarray
    score: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PreprocessResult:
    original: np.ndarray
    working: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    cleaned: np.ndarray
    scale: float
    debug_stages: dict[str, np.ndarray]
    selected_variant: str
    variant_scores: list[dict[str, float]]


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = pts.sum(axis=1)
    rect[0] = pts[np.argmin(summed)]
    rect[2] = pts[np.argmax(summed)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = max(1, int(round(max(width_a, width_b))))
    height = max(1, int(round(max(height_a, height_b))))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def try_perspective_correction(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if min(h, w) < 160:
        return gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    image_area = float(h * w)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18 or area > image_area * 0.95:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            LOGGER.debug("Applying perspective correction using contour area %.1f", area)
            return four_point_transform(gray, approx.reshape(4, 2).astype(np.float32))

    return gray


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)


def upscale_if_small(image: np.ndarray, min_side: int = 280) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    if side >= min_side:
        return image
    scale = min_side / max(1, side)
    size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def _open_binary(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _close_binary(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _binarize_adaptive(gray: np.ndarray, method: int, block_size: int, c_value: int) -> np.ndarray:
    block = max(11, block_size | 1)
    block = min(block, max(11, (min(gray.shape[:2]) // 2) * 2 - 1))
    return cv2.adaptiveThreshold(
        gray,
        255,
        method,
        cv2.THRESH_BINARY_INV,
        block,
        c_value,
    )


def _binarize_fixed(gray: np.ndarray, threshold: int) -> np.ndarray:
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _score_binary(binary: np.ndarray) -> tuple[float, dict[str, float]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return -1_000.0, {"components": 0.0, "plausible": 0.0, "median_area": 0.0}

    h, w = binary.shape[:2]
    image_area = float(h * w)
    min_area = max(8.0, image_area * 0.000015)

    areas_all = [float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels) if float(stats[label, cv2.CC_STAT_AREA]) >= min_area]
    if not areas_all:
        return -500.0, {
            "components": float(num_labels - 1),
            "plausible": 0.0,
            "median_area": 0.0,
            "noise": float(num_labels - 1),
            "large": 0.0,
        }

    all_array = np.array(areas_all, dtype=np.float32)
    median_area = float(np.median(all_array))
    plausible_areas = [area for area in areas_all if median_area * 0.35 <= area <= median_area * 2.5]
    noise_count = sum(1 for area in areas_all if area < median_area * 0.35)
    large_count = sum(1 for area in areas_all if area > median_area * 2.5)

    areas = np.array(plausible_areas or areas_all, dtype=np.float32)
    spread = float(np.std(areas) / max(median_area, 1.0))
    plausible = float(len(plausible_areas))
    density = plausible / max(1.0, image_area / 10_000.0)

    score = plausible * 5.0
    score += max(0.0, 30.0 - (spread * 18.0))
    score -= noise_count * 0.35
    score -= large_count * 8.0
    score -= max(0.0, density - 20.0) * 10.0

    return score, {
        "components": float(num_labels - 1),
        "plausible": plausible,
        "median_area": median_area,
        "spread": spread,
        "noise": float(noise_count),
        "large": float(large_count),
        "density": density,
    }


def generate_binary_variants(gray: np.ndarray) -> list[BinaryVariant]:
    h, w = gray.shape[:2]
    block = max(15, ((min(h, w) // 8) | 1))
    candidates: list[tuple[str, np.ndarray]] = [
        ("otsu_open", _open_binary(_binarize_otsu(gray))),
        ("otsu_open_close", _close_binary(_open_binary(_binarize_otsu(gray)))),
        ("adaptive_gaussian", _open_binary(_binarize_adaptive(gray, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block, 4))),
        ("adaptive_mean", _open_binary(_binarize_adaptive(gray, cv2.ADAPTIVE_THRESH_MEAN_C, block, 5))),
        ("fixed_200", _open_binary(_binarize_fixed(gray, 200))),
    ]

    variants: list[BinaryVariant] = []
    for name, image in candidates:
        score, metrics = _score_binary(image)
        variants.append(BinaryVariant(name=name, image=image, score=score, metrics=metrics))
    return variants


def select_best_binary(variants: list[BinaryVariant]) -> BinaryVariant:
    if not variants:
        raise ValueError("No binary variants supplied.")
    best = max(variants, key=lambda variant: variant.score)
    LOGGER.debug("Selected binary variant %s with score %.2f", best.name, best.score)
    return best


def preprocess_frame(image: np.ndarray, correct_perspective: bool = True) -> PreprocessResult:
    debug: dict[str, np.ndarray] = {}
    original = image.copy()
    working = upscale_if_small(original)
    scale = working.shape[0] / max(1, original.shape[0])
    debug["upscaled"] = working.copy()

    if working.ndim == 2:
        gray = working.copy()
        working_bgr = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        working_bgr = working.copy()
    debug["gray"] = gray.copy()

    corrected = try_perspective_correction(gray) if correct_perspective else gray
    debug["corrected"] = corrected.copy()

    enhanced = enhance_contrast(corrected)
    debug["enhanced"] = enhanced.copy()

    variants = generate_binary_variants(enhanced)
    best = select_best_binary(variants)
    debug["binary"] = best.image.copy()

    LOGGER.info(
        "Preprocessing selected %s (%s plausible components, median area %.1f)",
        best.name,
        int(best.metrics.get("plausible", 0)),
        best.metrics.get("median_area", 0.0),
    )

    return PreprocessResult(
        original=original,
        working=working_bgr,
        gray=gray,
        binary=best.image,
        cleaned=best.image,
        scale=scale,
        debug_stages=debug,
        selected_variant=best.name,
        variant_scores=[
            {"name": variant.name, "score": round(variant.score, 3), **variant.metrics}
            for variant in variants
        ],
    )


def resize_for_display(image: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / max(1, w)
    size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    ok, buffer = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Could not encode image.")
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_to_image(b64_string: str) -> np.ndarray:
    data = base64.b64decode(b64_string)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image.")
    return img
