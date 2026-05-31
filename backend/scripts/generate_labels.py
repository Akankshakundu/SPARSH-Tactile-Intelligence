"""
Auto-generate labels.csv from known sentences and detected cells.
"""
import json
import csv
from pathlib import Path

# Sentence-to-filename mapping (order matches braille1.png through braille20.png)
SENTENCES = [
    "pack my box with five dozen liquor jugs",
    "the quick brown fox jumps over the lazy dog",
    "how vexingly quick daft zebras jump",
    "that sentence has more than one thousand letters",
    "intense testing often happens in the evening",
    "about absolutely amazing animals always arrive",
    "the theory that trees teach trees to think",
    "sisters see seven seas in summer sunlight",
    "because better brains believe bigger possibilities",
    "in india intelligent insects investigate islands",
    "the rain in spain stays mainly in the plain",
    "every good boy deserves fruit daily",
    "bright yellow birds build tiny nest holes",
    "simple tests improve the recognition model",
    "walking slowly across the dark garden path",
    "ancient ruins hide secrets under moss and stone",
    "a small boat sailed across calm blue water",
    "five tiny kittens slept beside warm pillows",
    "the brave knight saved the injured traveler",
    "success comes after careful practice and effort",
]

# Load Braille cell definitions
braille_json_path = Path(__file__).parent.parent / "models" / "braille_cells.json"
with open(braille_json_path, "r", encoding="utf-8") as f:
    braille_data = json.load(f)

# Create pattern mapping
char_to_pattern = {}
for pattern, char in braille_data["letters"].items():
    char_to_pattern[char] = pattern

# Read samples.csv
samples_path = Path("../label_data/samples.csv")
samples = []
with open(samples_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    samples = list(reader)

# Group samples by source image
by_image = {}
for sample in samples:
    img = sample["source_image"]
    if img not in by_image:
        by_image[img] = []
    by_image[img].append(sample)

# Generate labels
labels = []
for idx, sentence in enumerate(SENTENCES):
    image_name = f"braille{idx+1}.png"
    print(f"\n{image_name}: {sentence}")
    
    if image_name not in by_image:
        print(f"  [ERROR] No detected cells for {image_name}")
        continue
    
    cells = sorted(by_image[image_name], key=lambda row: int(row["cell_index"]))
    cell_count = len(cells)
    print(f"  Detected {cell_count} cells")
    
    # Convert sentence to patterns (including spaces)
    patterns = []
    for char in sentence:
        if char == " ":
            patterns.append("000000")
        elif char.lower() in char_to_pattern:
            patterns.append(char_to_pattern[char.lower()])
        else:
            print(f"  [ERROR] Character '{char}' not in Braille mapping")
            patterns.append("000000")
    
    expected_count = len(patterns)
    print(f"  Sentence has {expected_count} characters (including spaces)")
    
    if cell_count != expected_count:
        print(f"  [ERROR] Detected cell count does not match sentence length for {image_name}.")
        print(f"          Detected={cell_count}, Expected={expected_count}")
        print("          This dataset is not safe to auto-label. Fix detection or use a different source.")
        raise SystemExit(1)
    
    # Assign patterns to cells in order
    for pattern_idx, (cell, pattern) in enumerate(zip(cells, patterns)):
        labels.append({
            "source_image": image_name,
            "cell_index": cell["cell_index"],
            "label_pattern": pattern,
        })
        print(f"    Cell {cell['cell_index']}: {pattern} ('{sentence[pattern_idx]}')")

# Write labels.csv
labels_path = Path("../label_data/labels.csv")
with open(labels_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["source_image", "cell_index", "label_pattern"])
    writer.writeheader()
    writer.writerows(labels)

print(f"\n✅ Generated {len(labels)} labels")
print(f"Saved to {labels_path}")
