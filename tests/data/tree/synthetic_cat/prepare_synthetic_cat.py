"""
Synthetic all-categorical classification datasets with a known generating rule,
to test categorical splits in the Tree environment.

Features (integer codes, independent and uniform):
    color: 4 categories (red=0, blue=1, green=2, yellow=3)
    size:  3 categories (small=0, medium=1, large=2)
    shape: 3 categories, pure noise in every dataset

Datasets (one label rule each, plus a fraction of flipped labels):
    cat_single:    y = 1 iff color == blue
        A single category: one subset split ``color in {blue}`` finds it.
        ``x <= t`` on the codes needs 2 splits, since blue (1) sits between
        red (0) and green (2).
    cat_subset:    y = 1 iff color in {red, green}
        Several categories: one subset split finds it, whereas a split that
        isolates a single category needs 2.
    cat_depth2:    y = 1 iff color == blue and size == small
        Needs a depth-2 tree; tests routing and ancestor constraints.

Output: ``<name>_<seed>.csv`` for seeds 1-5 with columns
``color, size, shape, class, Split`` (80/20 stratified train/test split), the
layout expected by ``Tree._load_dataset``. The files are written next to this
script, wherever it is run from:
    python tests/data/tree/synthetic_cat/prepare_synthetic_cat.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = Path(__file__).resolve().parent
N_CATEGORIES = {"color": 4, "size": 3, "shape": 3}


def make_features(n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {name: rng.integers(0, k, n_samples) for name, k in N_CATEGORIES.items()}
    )


def make_rules(X: pd.DataFrame) -> dict:
    return {
        "cat_single": X["color"] == 1,
        "cat_subset": X["color"].isin([0, 2]),
        "cat_depth2": (X["color"] == 1) & (X["size"] == 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    X = make_features(args.n_samples, rng)
    for name, rule in make_rules(X).items():
        y = rule.astype(int).to_numpy()
        # Label noise: flip a random fraction of the labels
        flip = rng.random(args.n_samples) < args.noise
        y[flip] = 1 - y[flip]
        for seed in args.seeds:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=args.test_size, random_state=seed
            )
            train_idx, test_idx = next(sss.split(X, y))
            df = pd.concat(
                [
                    X.iloc[train_idx].assign(**{"class": y[train_idx], "Split": "train"}),
                    X.iloc[test_idx].assign(**{"class": y[test_idx], "Split": "test"}),
                ],
                ignore_index=True,
            )
            df.to_csv(OUT_DIR / f"{name}_{seed}.csv", index=False)
        print(
            f"{name}: {args.n_samples} samples, {flip.sum()} labels flipped, "
            f"P(y=1) = {y.mean():.2f}"
        )


if __name__ == "__main__":
    main()
