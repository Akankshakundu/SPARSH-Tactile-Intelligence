"""Merge multiple .npz datasets for combined training.

Usage:
    python scripts/merge_datasets.py --datasets path1.npz path2.npz path3.npz --output combined.npz
"""

import argparse
import numpy as np
from pathlib import Path


def merge_datasets(dataset_paths: list[str], output_path: str) -> None:
    """Merge multiple .npz datasets into one."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_x = []
    all_y = []

    for path_str in dataset_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        data = np.load(path)
        if "x" not in data or "y" not in data:
            print(f"Skipping {path} (missing x or y)")
            continue

        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int32)

        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            print(f"Skipping {path} (invalid shape)")
            continue

        all_x.append(x)
        all_y.append(y)
        print(f"Loaded {len(x)} samples from {path}")

    if not all_x:
        raise ValueError("No valid datasets found to merge.")

    x_merged = np.vstack(all_x)
    y_merged = np.concatenate(all_y)

    np.savez_compressed(str(output_path), x=x_merged, y=y_merged)
    print(f"Merged into {len(x_merged)} total samples, saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple .npz datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Paths to .npz files to merge.")
    parser.add_argument("--output", required=True, help="Path to save the merged .npz.")
    args = parser.parse_args()

    merge_datasets(args.datasets, args.output)


if __name__ == "__main__":
    main()
