"""
Image preprocessing pipeline for Braille recognition.
Handles upscale for small uploads, multi-pass binarization, safe perspective correction,
contrast enhancement, and preparation for dot detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    original: np.ndarray
    working: np.ndarray  # upscaled BGR used for detection + overlay (same coords as cleaned)
    gray: np.ndarray
    binary: np.ndarray
    cleaned: np.ndarray
    scale: float
    debug_stages: dict


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (maxWidth, maxHeight))


def try_perspective_correction(gray: np.ndarray) -> np.ndarray:
    """
    Correct document tilt only when a clear inner quadrilateral is found.
    Skips tiny images and full-frame borders (common on white PNG crops).
    """
    h, w = gray.shape[:2]
    if min(h, w) < 120:
        return gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return gray

    image_area = float(h * w)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * 0.12 or area > image_area * 0.92:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            return four_point_transform(gray, pts)

    return gray


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _binarize_otsu(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return binary


def _binarize_adaptive(gray: np.ndarray, block_size: int = 31) -> np.ndarray:
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = max(11, min(block, min(gray.shape[:2]) // 2 * 2 - 1))
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=4,
    )


def _binarize_fixed(gray: np.ndarray) -> np.ndarray:
    """Good for faint black dots on near-white digital PNGs."""
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary


def upscale_if_small(image: np.ndarray, min_side: int = 280) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    if side >= min_side:
        return image
    scale = min_side / side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def remove_noise(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def generate_binary_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Several binarization strategies; dot_detector picks the richest result."""
    variants: list[np.ndarray] = []
    std = float(np.std(gray))
    h, w = gray.shape[:2]
    block = max(11, min(31, (min(h, w) // 8) | 1))

    variants.append(remove_noise(_binarize_otsu(gray, invert=True)))
    variants.append(remove_noise(_binarize_adaptive(gray, block)))
    variants.append(remove_noise(_binarize_fixed(gray)))

    if std < 50:
        variants.append(remove_noise(_binarize_otsu(gray, invert=False)))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(remove_noise(_binarize_otsu(blurred, invert=True)))

    return variants


def select_best_binary(gray: np.ndarray, variants: list[np.ndarray]) -> np.ndarray:
    """Choose the variant that yields the most plausible dot blobs."""
    from .dot_detector import count_dot_candidates

    best = variants[0]
    best_score = -1
    for binary in variants:
        score = count_dot_candidates(binary)
        if score > best_score:
            best_score = score
            best = binary
    return best


def preprocess_frame(image: np.ndarray, correct_perspective: bool = True) -> PreprocessResult:
    debug: dict = {}
    h0, w0 = image.shape[:2]
    working = upscale_if_small(image)
    h1, w1 = working.shape[:2]
    scale = h1 / h0 if h0 else 1.0
    debug["upscaled"] = working.copy()

    if len(working.shape) == 3:
        working_bgr = working
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    else:
        working_bgr = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        gray = working.copy()
    debug["gray"] = gray.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Small / digital crops: perspective warp often destroys layout
    use_perspective = correct_perspective and min(gray.shape[:2]) >= 160
    if use_perspective:
        corrected = try_perspective_correction(blurred)
    else:
        corrected = blurred
    debug["corrected"] = corrected.copy()

    enhanced = enhance_contrast(corrected)
    debug["enhanced"] = enhanced.copy()

    variants = generate_binary_variants(enhanced)
    cleaned = select_best_binary(enhanced, variants)
    debug["binary"] = cleaned.copy()

    return PreprocessResult(
        original=image,
        working=working_bgr,
        gray=gray,
        binary=cleaned,
        cleaned=cleaned,
        scale=scale,
        debug_stages=debug,
    )


def resize_for_display(image: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    import base64

    _, buffer = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_to_image(b64_string: str) -> np.ndarray:
    import base64

    data = base64.b64decode(b64_string)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
