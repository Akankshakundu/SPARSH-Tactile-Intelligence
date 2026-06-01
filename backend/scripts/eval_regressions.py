"""Measure exact-match and character accuracy across regression cases."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ml_model import initialize_ml_model
from core.recognition import run_recognition

DATA = json.load(
    open(os.path.join(os.path.dirname(__file__), "..", "models", "braille_cells.json"), encoding="utf-8")
)
LETTERS = DATA["letters"]
PUNCT = DATA["punctuation"]

MESSAGE = list("you can do it !")
PATTERNS = []
for ch in MESSAGE:
    if ch == " ":
        PATTERNS.append("000000")
    elif ch in LETTERS.values():
        for pat, letter in LETTERS.items():
            if letter == ch:
                PATTERNS.append(pat)
                break
    elif ch == "!":
        PATTERNS.append("001111")


def draw_braille_line(patterns, dot_r=8, cell_gap=28, dot_gap=10, word_gap=52):
    h = 100
    w = len(patterns) * (dot_r * 6 + cell_gap) + 40
    img = np.ones((h, w), dtype=np.uint8) * 255
    x0 = 20
    row_y = [18, 18 + dot_gap * 2, 18 + dot_gap * 4]
    col_dx = [0, dot_gap * 2]
    for pat in patterns:
        if pat == "000000":
            x0 += word_gap
            continue
        for i, bit in enumerate(pat):
            if bit != "1":
                continue
            col, row = (0, i) if i < 3 else (1, i - 3)
            cv2.circle(img, (int(x0 + col_dx[col]), int(row_y[row])), dot_r, 0, -1)
        x0 += dot_r * 6 + cell_gap
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_tiny_white_cat():
    patterns = []
    for ch in "cat":
        for pat, letter in LETTERS.items():
            if letter == ch:
                patterns.append(pat)
                break
    big = draw_braille_line(patterns, dot_r=10, cell_gap=36, dot_gap=12)
    return cv2.resize(big, (120, 40), interpolation=cv2.INTER_AREA)


def draw_sentence_strip(text: str, word_gap: int = 70):
    patterns = []
    for ch in text:
        if ch == " ":
            patterns.append("000000")
        elif ch in LETTERS.values():
            for pat, letter in LETTERS.items():
                if letter == ch:
                    patterns.append(pat)
                    break
    return draw_braille_line(patterns, dot_r=9, cell_gap=30, dot_gap=11, word_gap=word_gap)


CASES = [
    ("synthetic: you can do it !", draw_braille_line(PATTERNS), "you can do it !", False),
    ("synthetic: this is a great city", draw_sentence_strip("this is a great city", 72), "this is a great city", False),
    ("synthetic: tiny cat (perspective=True)", draw_tiny_white_cat(), "cat", True),
    ("synthetic: tiny cat (perspective=False)", draw_tiny_white_cat(), "cat", False),
]

_root = os.path.join(os.path.dirname(__file__), "..", "..")
for name in ("abc", "hello", "test_braille", "i_love_you", "good_morning"):
    path = os.path.join(_root, "sample_inputs", f"{name}.png")
    if os.path.isfile(path):
        text = name.replace("_", " ") if name != "test_braille" else "braille"
        CASES.append((f"sample_inputs/{name}.png", cv2.imread(path), text, False))


def char_accuracy(expected: str, actual: str) -> float:
    exp = expected.replace(" ", "")
    act = actual.replace(" ", "")
    if not exp:
        return 1.0 if not act else 0.0
    matches = sum(1 for a, b in zip(exp, act) if a == b)
    return matches / max(len(exp), len(act))


def evaluate():
    results = []
    exact = 0
    char_total = 0.0
    for name, img, expected, persp in CASES:
        if img is None:
            continue
        r = run_recognition(img, correct_perspective=persp)
        ok = r.text == expected
        ca = char_accuracy(expected, r.text)
        exact += int(ok)
        char_total += ca
        results.append({"input": name, "expected": expected, "output": r.text, "exact_match": ok, "char_accuracy": round(ca, 4)})
    n = len(results)
    return {"exact_matches": exact, "total_tests": n, "mean_char_accuracy": round(char_total / n, 4) if n else 0.0, "results": results}


if __name__ == "__main__":
    initialize_ml_model(force_synthetic=False)
    print(json.dumps(evaluate(), indent=2))
