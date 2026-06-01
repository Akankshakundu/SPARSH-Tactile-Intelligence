"""Train and persist the Braille cell classifier.

Usage:
    python train_model.py --output ../models/ml_model.npz
    python train_model.py --dataset dataset.npz --output ../models/ml_model.npz
    python train_model.py --dataset dataset.npz --augment-synthetic --output ../models/ml_model.npz

The dataset should be a NumPy .npz archive with arrays named `x` and `y`.
`x` should be shape (N, 6) and `y` should be shape (N,) with integer class ids.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ml_model import generate_synthetic_dataset, train_on_dataset, save_trained_model


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "x" not in data or "y" not in data:
        raise ValueError("Dataset archive must contain 'x' and 'y' arrays.")
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int32)
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("x must be shape (N, 6).")
    if y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("y must be shape (N,) and match x length.")
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Braille cell classifier from real or synthetic data.")
    parser.add_argument("--dataset", help="Path to a .npz dataset file with x and y arrays.")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "models", "ml_model.npz"), help="Path to save the trained model.")
    parser.add_argument("--samples-per-class", type=int, default=120, help="Synthetic samples per class when generating baseline data.")
    parser.add_argument("--augment-synthetic", action="store_true", help="Augment provided dataset with synthetic examples.")
    args = parser.parse_args()

    if args.dataset:
        print(f"Loading training dataset from {args.dataset}")
        x, y = load_dataset(args.dataset)
        print(f"Loaded {len(x)} examples from dataset.")
        if args.augment_synthetic:
            xs, ys = generate_synthetic_dataset(samples_per_class=args.samples_per_class)
            x = np.vstack([x, xs])
            y = np.concatenate([y, ys])
            print(f"Augmented with {len(xs)} synthetic examples, total={len(x)}.")
    else:
        print("No dataset supplied. Generating synthetic training data only.")
        x, y = generate_synthetic_dataset(samples_per_class=args.samples_per_class)
        print(f"Generated {len(x)} synthetic examples.")

    train_on_dataset(x, y, save=True, path=args.output)
    print(f"Saved trained model to {args.output}")
