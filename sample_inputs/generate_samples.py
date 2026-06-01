"""
Generate synthetic Braille sample images for testing inference.py.
Run this once to populate sample_inputs/ with test images.

Usage:
    python sample_inputs/generate_samples.py
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

import cv2
import numpy as np

DATA_PATH = os.path.join(BACKEND_DIR, "models", "braille_cells.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

LETTERS = DATA["letters"]
PUNCT = DATA["punctuation"]

OUT_DIR = SCRIPT_DIR


def draw_braille_line(
    patterns: list[str],
    dot_r: int = 9,
    cell_gap: int = 30,
    dot_gap: int = 11,
    word_gap: int = 60,
    bg: int = 255,
    fg: int = 0,
) -> np.ndarray:
    h = 100
    w = len(patterns) * (dot_r * 6 + cell_gap) + 80
    img = np.ones((h, w), dtype=np.uint8) * bg

    x0 = 30
    row_y = [18, 18 + dot_gap * 2, 18 + dot_gap * 4]
    col_dx = [0, dot_gap * 2]

    for pat in patterns:
        if pat == "000000":
            x0 += word_gap
            continue
        for i, bit in enumerate(pat):
            if bit != "1":
                continue
            col = 0 if i < 3 else 1
            row = i if i < 3 else i - 3
            cx = x0 + col_dx[col]
            cy = row_y[row]
            cv2.circle(img, (cx, cy), dot_r, fg, -1)
        x0 += dot_r * 6 + cell_gap

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def text_to_patterns(text: str) -> list[str]:
    patterns = []
    for ch in text.lower():
        if ch == " ":
            patterns.append("000000")
        else:
            for pat, letter in LETTERS.items():
                if letter == ch:
                    patterns.append(pat)
                    break
            else:
                for pat, sym in PUNCT.items():
                    if sym == ch:
                        patterns.append(pat)
                        break
    return patterns


SAMPLES = [
    ("hello", "hello"),
    ("test_braille", "braille"),
    ("i_love_you", "i love you"),
    ("good_morning", "good morning"),
    ("abc", "abc"),
]

for filename, text in SAMPLES:
    patterns = text_to_patterns(text)
    # Use same params as test_pipeline.py for reliable detection
    img = draw_braille_line(patterns, dot_r=8, cell_gap=28, dot_gap=10, word_gap=52)
    out_path = os.path.join(OUT_DIR, f"{filename}.png")
    cv2.imwrite(out_path, img)
    print(f"Generated: {out_path}  (text: '{text}')")

print(f"\nDone. {len(SAMPLES)} sample images saved to {OUT_DIR}")
print("Run inference with:")
print("  python inference.py --source sample_inputs/")
