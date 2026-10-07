"""Shared paths, splits and metrics for the re-run of the COMP9517 experiments.

Protocol (fixed for every model and setting):
  * test set   : data/Aerial_Landscapes_Test, 160 images per class, touched once at the end
  * val set    : 96 images per class carved from data/Aerial_Landscapes_Train (seed 0);
                 identical in all settings, used for every hyperparameter / early-stopping decision
  * train pool : the remaining 544 images per class
  * settings   : balanced   - full train pool
                 longtail   - per-class subsample of the pool, ratios 1.0 ... 0.0625 (imbalance 16x)
                 rebalanced - longtail + offline augmentation of minority classes back to 544 each
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
SPLITS = RUNS / "splits"

TRAIN_DIR = DATA / "Aerial_Landscapes_Train"
TEST_DIR = DATA / "Aerial_Landscapes_Test"
AUG_DIR = DATA / "derived" / "rebalanced_aug"

SEED = 0
VAL_PER_CLASS = 96
# Original long-tail targets (800 ... 50 out of 800); applied as ratios to the 544-image pool.
LT_TARGETS = [800, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
SETTINGS = ("balanced", "longtail", "rebalanced")

CLASSES = sorted(d for d in os.listdir(TRAIN_DIR) if not d.startswith("."))
NUM_CLASSES = len(CLASSES)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _list(d: Path) -> list[tuple[str, int]]:
    out = []
    for i, c in enumerate(CLASSES):
        for f in sorted(os.listdir(d / c)):
            if not f.startswith("."):
                out.append((str((d / c / f).relative_to(ROOT)), i))
    return out


def augment_image(img, rng: random.Random):
    """Offline augmentation used for rebalancing (unchanged from the original notebook)."""
    from PIL import ImageEnhance, ImageFilter

    img = img.rotate(rng.uniform(-30, 30))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 2.0)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    return img


def build_splits() -> None:
    """Write runs/splits/<setting>.json with train/val/test (path, label) lists."""
    from PIL import Image

    SPLITS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pool = _list(TRAIN_DIR)
    test = _list(TEST_DIR)

    by_cls = {i: [p for p in pool if p[1] == i] for i in range(NUM_CLASSES)}
    val, train_bal = [], []
    for i, items in by_cls.items():
        idx = rng.permutation(len(items))
        val += [items[j] for j in idx[:VAL_PER_CLASS]]
        train_bal += [items[j] for j in idx[VAL_PER_CLASS:]]

    n_pool = len(by_cls[0]) - VAL_PER_CLASS
    lt_counts = [round(n_pool * t / LT_TARGETS[0]) for t in LT_TARGETS]
    train_lt = []
    for i in range(NUM_CLASSES):
        items = [p for p in train_bal if p[1] == i]
        idx = rng.permutation(len(items))[: lt_counts[i]]
        train_lt += [items[j] for j in sorted(idx)]

    # Rebalanced: augment each minority class back up to n_pool using its own long-tail images only.
    train_rb = list(train_lt)
    prng = random.Random(SEED)
    for i in range(NUM_CLASSES):
        own = [p for p in train_lt if p[1] == i]
        out_dir = AUG_DIR / CLASSES[i]
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_pool - len(own)):
            src, _ = own[k % len(own)]
            dst = out_dir / f"{Path(src).stem}_aug{k}.jpg"
            if not dst.exists():
                augment_image(Image.open(ROOT / src).convert("RGB"), prng).save(dst, quality=95)
            train_rb.append((str(dst.relative_to(ROOT)), i))

    for name, tr in (("balanced", train_bal), ("longtail", train_lt), ("rebalanced", train_rb)):
        counts = np.bincount([y for _, y in tr], minlength=NUM_CLASSES).tolist()
        json.dump(
            {"classes": CLASSES, "train_counts": counts, "train": tr, "val": val, "test": test},
            open(SPLITS / f"{name}.json", "w"),
        )
        print(f"{name:10s} train={len(tr):5d} val={len(val)} test={len(test)} counts={counts}")


def load_split(setting: str) -> dict:
    return json.load(open(SPLITS / f"{setting}.json"))


def macro_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred), "prec": p, "rec": r, "f1": f}


def save_outputs(setting: str, model: str, val_prob, test_prob, meta: dict) -> None:
    d = RUNS / setting / model
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "val_prob.npy", np.asarray(val_prob, dtype=np.float32))
    np.save(d / "test_prob.npy", np.asarray(test_prob, dtype=np.float32))
    json.dump(meta, open(d / "meta.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    build_splits()
