"""
Sparsh Tactile ML Engine — hybrid classifier with honest confidence scores.
"""

import json
import os

import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "braille_cells.json")
_MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "ml_model.npz")

with open(_DATA_PATH, "r", encoding="utf-8") as f:
    _BRAILLE_DATA = json.load(f)

_LETTERS = _BRAILLE_DATA["letters"]
_PUNCTUATION = {k: v for k, v in _BRAILLE_DATA["punctuation"].items() if v not in ("capital", "number")}
_ALL_PATTERNS: dict[str, str] = {}
_ALL_PATTERNS.update(_LETTERS)
_ALL_PATTERNS.update(_PUNCTUATION)
_ALL_PATTERNS.update(_BRAILLE_DATA.get("contractions", {}))

_CLASSES = list(_LETTERS.keys()) + list(_PUNCTUATION.keys())
_CLASS_MAP = {pat: idx for idx, pat in enumerate(_CLASSES)}
_INV_CLASS_MAP = {idx: pat for pat, idx in _CLASS_MAP.items()}


def pattern_to_display_char(pattern: str) -> str:
    if pattern == "000000":
        return "·"
    if pattern in _LETTERS:
        return _LETTERS[pattern]
    if pattern in _PUNCTUATION:
        ch = _PUNCTUATION[pattern]
        return ch if len(ch) == 1 else ch
    return "?"


class TactileKNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train = x
        self.y_train = y

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        if self.x_train is None or self.y_train is None:
            bits = "".join("1" if val > 0.5 else "0" for val in feature_vector)
            return bits, 0.5

        distances = np.linalg.norm(self.x_train - feature_vector, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_classes = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        counts = np.bincount(k_nearest_classes)
        pred_class_idx = int(np.argmax(counts))

        best_dist = float(k_nearest_distances[0])
        knn_conf = float(np.exp(-best_dist * 2.2))

        pred_pattern = _INV_CLASS_MAP.get(pred_class_idx, "000000")
        return pred_pattern, max(0.15, min(0.92, knn_conf))


_model_instance = TactileKNNClassifier(k=5)


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _match_by_hamming(binary_pattern: str, max_distance: int = 1) -> tuple[str | None, float]:
    best_pat = None
    best_dist = max_distance + 1
    for pat in _CLASSES:
        d = _hamming(binary_pattern, pat)
        if d < best_dist:
            best_dist = d
            best_pat = pat
    if best_pat is None or best_dist > max_distance:
        return None, 0.0
    return best_pat, max(0.45, 1.0 - (best_dist / 6.0) * 0.55)


def _geometric_confidence(binary_pattern: str, intensities: list[float] | None) -> float:
    """How well detected dots match the binary pattern (0–1)."""
    if not intensities or len(intensities) != 6:
        return 0.65

    raised = [i for i, b in enumerate(binary_pattern) if b == "1"]
    if not raised:
        return 0.7 if max(intensities) < 0.35 else 0.4

    hit_scores = [intensities[i] for i in raised]
    hit_mean = float(np.mean(hit_scores))

    false_pos = sum(1 for i, v in enumerate(intensities) if binary_pattern[i] == "0" and v > 0.5)
    penalty = false_pos * 0.12

    return max(0.25, min(1.0, hit_mean - penalty))


def _blend_confidence(
    classifier_conf: float,
    binary_pattern: str,
    predicted_pattern: str,
    intensities: list[float] | None,
) -> float:
    geom = _geometric_confidence(binary_pattern, intensities)
    hamming_pen = _hamming(binary_pattern, predicted_pattern) * 0.09
    blended = classifier_conf * 0.45 + geom * 0.55 - hamming_pen
    return round(max(0.22, min(0.97, blended)), 3)


def classify_cell(
    binary_pattern: str,
    intensity_vector: list[float] | None = None,
) -> tuple[str, str, float]:

    
    if len(binary_pattern) != 6 or not all(c in "01" for c in binary_pattern):
        binary_pattern = "".join("1" if v > 0.5 else "0" for v in (intensity_vector or [0] * 6))

    if binary_pattern in _ALL_PATTERNS:
        conf = _blend_confidence(0.88, binary_pattern, binary_pattern, intensity_vector)
        return binary_pattern, pattern_to_display_char(binary_pattern), conf

    near, near_conf = _match_by_hamming(
    binary_pattern,
    max_distance=1
    )

    if near and _geometric_confidence(
        binary_pattern,
        intensity_vector
    ) >= 0.55:

        conf = _blend_confidence(
            near_conf,
            binary_pattern,
            near,
            intensity_vector
        )

        return (
            near,
            pattern_to_display_char(near),
            conf
        )

    if intensity_vector and len(intensity_vector) == 6:
        feature = np.array(intensity_vector, dtype=np.float32)
    else:
        feature = np.array([float(b) for b in binary_pattern], dtype=np.float32)

    pred_pattern, knn_conf = _model_instance.predict(feature)
    conf = _blend_confidence(knn_conf, binary_pattern, pred_pattern, intensity_vector)
    return pred_pattern, pattern_to_display_char(pred_pattern), conf


def generate_synthetic_dataset(samples_per_class: int = 150) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[int] = []

    np.random.seed(42)

    for pattern, class_idx in _CLASS_MAP.items():
        ideal = np.array([float(bit) for bit in pattern])
        for _ in range(samples_per_class):
            noise = np.random.normal(0, 0.07, size=6)
            distorted = ideal + noise
            if np.random.random() < 0.18:
                raised = [i for i, b in enumerate(pattern) if b == "1"]
                if raised:
                    distorted[raised[np.random.randint(len(raised))]] *= np.random.uniform(0.25, 0.65)
            shadow = np.linspace(1.0, np.random.uniform(0.7, 0.95), 6)
            distorted = np.clip(distorted * shadow, 0.0, 1.0)
            x_data.append(distorted)
            y_data.append(class_idx)

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int32)


def save_trained_model(x: np.ndarray, y: np.ndarray, path: str | None = None) -> None:
    path = path or _MODEL_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.int32))


def load_trained_model(path: str | None = None) -> bool:
    path = path or _MODEL_FILE
    if not os.path.exists(path):
        return False

    data = np.load(path)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int32)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        return False
    _model_instance.fit(x, y)
    return True


def train_on_dataset(x: np.ndarray, y: np.ndarray, save: bool = True, path: str | None = None) -> None:
    x = x.astype(np.float32)
    y = y.astype(np.int32)
    _model_instance.fit(x, y)
    if save:
        save_trained_model(x, y, path)


def initialize_ml_model(force_synthetic: bool = False) -> None:
    if not force_synthetic and load_trained_model():
        print(f"[SPARSH ML] Loaded saved model from {_MODEL_FILE}")
        return

    print("[SPARSH ML] Generating augmented training dataset...")
    x, y = generate_synthetic_dataset()
    print(f"[SPARSH ML] Training KNN on {len(x)} vectors ({len(_CLASSES)} classes)...")
    _model_instance.fit(x, y)
    print("[SPARSH ML] Model ready.")


def predict_cell(dots_intensity: list[float]) -> tuple[str, float]:
    binary = "".join("1" if v > 0.45 else "0" for v in dots_intensity)
    pattern, _, conf = classify_cell(binary, dots_intensity)
    return pattern, conf
