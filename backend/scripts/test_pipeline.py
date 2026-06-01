"""Quick offline test: synthetic Braille image + recognition pipeline."""
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

# "you can do it !"
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
            cx = x0 + col_dx[col]
            cy = row_y[row]
            cv2.circle(img, (cx, cy), dot_r, 0, -1)
        x0 += dot_r * 6 + cell_gap

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_tiny_white_cat():
    """Simulates a small white PNG like Untitled1.png (low resolution)."""
    patterns = []
    for ch in "cat":
        for pat, letter in LETTERS.items():
            if letter == ch:
                patterns.append(pat)
                break
    big = draw_braille_line(patterns, dot_r=10, cell_gap=36, dot_gap=12)
    # Downscale to mimic 1KB upload
    small = cv2.resize(big, (120, 40), interpolation=cv2.INTER_AREA)
    return small


def draw_sentence_strip(text: str, word_gap: int = 70) -> np.ndarray:
    """Wide strip like yellow training images with clear word gaps."""
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


def draw_sentence(text: str) -> np.ndarray:
    patterns = []
    for ch in text:
        if ch == " ":
            patterns.append("000000")
        elif ch in LETTERS.values():
            for pat, letter in LETTERS.items():
                if letter == ch:
                    patterns.append(pat)
                    break
        else:
            for pat, sym in PUNCT.items():
                if sym == ch:
                    patterns.append(pat)
                    break
    return draw_braille_line(patterns, dot_r=9, cell_gap=32, dot_gap=11)


if __name__ == "__main__":
    initialize_ml_model()

    print("=== Full sentence test ===")
    img = draw_braille_line(PATTERNS)
    result = run_recognition(img, correct_perspective=False)
    print("Expected:", "".join(MESSAGE))
    print("Decoded: ", result.text)
    print("Cells:", result.cell_count, "Confidence:", result.confidence)

    print("\n=== 'this is a great city' (strip) ===")
    city = draw_sentence_strip("this is a great city", word_gap=72)
    r2 = run_recognition(city, correct_perspective=False)
    print("Expected: this is a great city")
    print("Decoded: ", r2.text)
    print("Cells:", r2.cell_count)
    for c in r2.cells:
        ch = c["char"]
        if ch != " ":
            print(f"  {ch} {c['confidence_pct']}%")
        else:
            print("  [space]")

    print("\n=== Tiny white 'cat' (simulates Untitled1.png) ===")
    tiny = draw_tiny_white_cat()
    for persp in (True, False):
        r = run_recognition(tiny, correct_perspective=persp)
        print(f"perspective={persp} -> {repr(r.text)} cells={r.cell_count} dots={r.dot_count}")
        for c in r.cells:
            print(f"  {c['char']} {c['confidence_pct']}%")
