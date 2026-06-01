"""
Create a review labels CSV for a single source image using extracted samples.
Usage:
  python scripts/make_review_labels.py --image braille8.png --out ../label_data/labels_review_braille8.csv
The output CSV will contain rows: source_image,cell_index,label_pattern (empty)
"""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Source image filename (e.g. braille8.png)")
parser.add_argument("--out", required=True, help="Output CSV path")
args = parser.parse_args()

samples_path = Path(__file__).parent.parent / "label_data" / "samples.csv"
out_path = Path(args.out)

rows = []
with samples_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["source_image"] == args.image:
            rows.append({
                "source_image": r["source_image"],
                "cell_index": r["cell_index"],
                "label_pattern": "",
            })

if not rows:
    print(f"No samples found for {args.image} in {samples_path}")
else:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_image", "cell_index", "label_pattern"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} review rows to {out_path}")
