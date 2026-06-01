import os
import json
import csv
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BRAILLE_JSON = os.path.join(BASE_DIR, "..", "braille_cells.json")
SENTENCES_FILE = os.path.join(BASE_DIR, "sentences.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_images")
LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")

DOT_RADIUS = 6
DOT_SPACING_X = 18
DOT_SPACING_Y = 18

CELL_SPACING = 30
LINE_SPACING = 50

MARGIN = 20

# =========================
# LOAD BRAILLE MAPPING
# =========================

with open(BRAILLE_JSON, "r", encoding="utf-8") as f:
    braille_data = json.load(f)

char_to_pattern = {}

# Letters
for pattern, char in braille_data["letters"].items():
    char_to_pattern[char.lower()] = pattern

# Numbers
for pattern, char in braille_data["numbers"].items():
    if char != "#":
        char_to_pattern[char] = pattern

# Punctuation
for pattern, char in braille_data["punctuation"].items():
    if len(char) == 1:
        char_to_pattern[char] = pattern

# Space handling
SPACE_WIDTH = CELL_SPACING * 2


# =========================
# DRAW ONE BRAILLE CELL
# =========================

def draw_braille_cell(img, pattern, x0, y0):
    """
    Pattern example:
    100000

    Layout:

    1 4
    2 5
    3 6
    """

    positions = [
        (0, 0),  # 1
        (0, 1),  # 2
        (0, 2),  # 3
        (1, 0),  # 4
        (1, 1),  # 5
        (1, 2),  # 6
    ]

    for idx, bit in enumerate(pattern):
        if bit == "1":
            col, row = positions[idx]

            cx = x0 + col * DOT_SPACING_X
            cy = y0 + row * DOT_SPACING_Y

            cv2.circle(
                img,
                (int(cx), int(cy)),
                DOT_RADIUS,
                (0, 0, 0),
                -1
            )


# =========================
# TEXT -> BRAILLE IMAGE
# =========================

def text_to_braille_image(text):
    lines = text.strip().split("\n")

    max_chars = max(len(line) for line in lines)

    width = (
        MARGIN * 2
        + max_chars * CELL_SPACING
        + 100
    )

    height = (
        MARGIN * 2
        + len(lines) * LINE_SPACING
        + 100
    )

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for line_idx, line in enumerate(lines):

        x = MARGIN
        y = MARGIN + line_idx * LINE_SPACING + 20

        for ch in line.lower():

            if ch == " ":
                x += SPACE_WIDTH
                continue

            if ch not in char_to_pattern:
                x += CELL_SPACING
                continue

            pattern = char_to_pattern[ch]

            draw_braille_cell(
                img,
                pattern,
                x,
                y
            )

            x += CELL_SPACING

    return img


# =========================
# LOAD SENTENCES
# =========================

with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
    texts = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

# =========================
# OUTPUT FOLDER
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# GENERATE IMAGES
# =========================

rows = []

for idx, text in enumerate(texts, start=1):

    img = text_to_braille_image(text)

    filename = f"sample_{idx:04d}.png"

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(filepath, img)

    rows.append([filename, text])

# =========================
# SAVE LABELS CSV
# =========================

with open(
    LABELS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "text"
    ])

    writer.writerows(rows)

print(f"Generated {len(rows)} images.")
print(f"Images saved in: {OUTPUT_DIR}")
print(f"Labels saved in: {LABELS_CSV}")
