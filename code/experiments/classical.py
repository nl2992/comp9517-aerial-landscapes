"""Hand-crafted feature pipelines: LBP-kNN, colour-histogram SVM, SIFT-BoW SVM, stacking.

Every hyperparameter is chosen by Optuna (TPE) on the shared balanced validation set;
the test set is only scored once with the selected configuration.

    python code/experiments/classical.py [setting ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per joblib worker; avoids oversubscription

import sys
import time

import cv2
import numpy as np
import optuna
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import ROOT, SEED, SETTINGS, load_split, macro_metrics, save_outputs, seed_everything

LBP_P, LBP_R = 24, 3  # uniform LBP -> P + 2 = 26 bins
COLOUR_BINS = 32  # per RGB channel -> 96 dims
VOCAB = 100  # SIFT visual words
SIFT_MAX = 300  # keypoints per image
KMEANS_SAMPLE = 200_000
N_JOBS = int(os.environ.get("N_JOBS", 6))  # leave cores for the CNN queue

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _extract(path: str):
    cv2.setNumThreads(1)
    img = cv2.imread(str(ROOT / path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    h_lbp = np.bincount(lbp.astype(int).ravel(), minlength=LBP_P + 2).astype(np.float32)
    h_lbp /= h_lbp.sum()
    h_col = np.hstack(
        [np.bincount((img[..., c] // (256 // COLOUR_BINS)).ravel(), minlength=COLOUR_BINS) for c in (2, 1, 0)]
    ).astype(np.float32)
    h_col /= h_col.sum() / 3.0  # each channel sums to 1
    _, des = cv2.SIFT_create(nfeatures=SIFT_MAX).detectAndCompute(gray, None)
    des = np.zeros((0, 128), np.uint8) if des is None else np.clip(des, 0, 255).astype(np.uint8)
    return h_lbp, h_col, des


def extract_all(paths: list[str]) -> dict:
    t = time.time()
    out = Parallel(n_jobs=N_JOBS, batch_size=64)(delayed(_extract)(p) for p in paths)
    print(f"  extracted {len(paths)} images in {time.time() - t:.0f}s", flush=True)
    return dict(zip(paths, out))


def bow(feats: dict, train_paths: list[str], paths_by_split: dict) -> dict:
    rng = np.random.default_rng(SEED)
    des = np.vstack([feats[p][2] for p in train_paths if len(feats[p][2])])
    sample = des[rng.choice(len(des), min(KMEANS_SAMPLE, len(des)), replace=False)].astype(np.float32)
    km = MiniBatchKMeans(n_clusters=VOCAB, batch_size=4096, random_state=SEED, n_init=3).fit(sample)
    out = {}
    for split, paths in paths_by_split.items():
        H = np.zeros((len(paths), VOCAB), np.float32)
        for i, p in enumerate(paths):
            d = feats[p][2]
            if len(d):
                H[i] = np.bincount(km.predict(d.astype(np.float32)), minlength=VOCAB)
                H[i] /= H[i].sum()
        out[split] = H
    return out


def tune(make, Xtr, ytr, Xva, yva, space, n_trials):
    def objective(trial):
        model = make(space(trial)).fit(Xtr, ytr)
        return f1_score(yva, model.predict(Xva), average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def svm(p, prob=False):
    return make_pipeline(StandardScaler(), SVC(C=p["C"], gamma=p["gamma"], probability=prob, random_state=SEED))


def svm_space(t):
    return {"C": t.suggest_float("C", 1e-2, 1e3, log=True), "gamma": t.suggest_float("gamma", 1e-4, 1.0, log=True)}


def knn(p):
    return KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"], metric=p["metric"])


def knn_space(t):
    return {
        "k": t.suggest_int("k", 1, 30),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": t.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }


def stack(p):
    return make_pipeline(
        StandardScaler(),
        StackingClassifier(
            estimators=[
                ("svm", SVC(C=p["C"], gamma="scale", random_state=SEED)),
                ("knn", KNeighborsClassifier(n_neighbors=p["k"], weights="distance")),
                ("dt", DecisionTreeClassifier(max_depth=p["depth"], random_state=SEED)),
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=5,
            n_jobs=N_JOBS,
        ),
    )


def stack_space(t):
    return {
        "C": t.suggest_float("C", 1e-1, 1e2, log=True),
        "k": t.suggest_int("k", 1, 30),
        "depth": t.suggest_int("depth", 2, 40),
    }


def run_setting(setting: str, feats: dict) -> None:
    s = load_split(setting)
    paths = {k: [p for p, _ in s[k]] for k in ("train", "val", "test")}
    y = {k: np.array([c for _, c in s[k]]) for k in ("train", "val", "test")}
    lbp = {k: np.stack([feats[p][0] for p in v]) for k, v in paths.items()}
    col = {k: np.stack([feats[p][1] for p in v]) for k, v in paths.items()}
    bw = bow(feats, paths["train"], paths)
    comb = {k: np.hstack([lbp[k], col[k], bw[k]]) for k in paths}

    jobs = [
        ("lbp_knn", lbp, knn, knn_space, 30, lambda p: knn(p)),
        ("colour_svm", col, svm, svm_space, 30, lambda p: svm(p, prob=True)),
        ("sift_svm", bw, svm, svm_space, 30, lambda p: svm(p, prob=True)),
        ("stacking", comb, stack, stack_space, 15, stack),
    ]
    for name, X, make, space, n_trials, final in jobs:
        t = time.time()
        best, val_f1 = tune(make, X["train"], y["train"], X["val"], y["val"], space, n_trials)
        model = final(best).fit(X["train"], y["train"])
        pv, pt = model.predict_proba(X["val"]), model.predict_proba(X["test"])
        m = macro_metrics(y["test"], pt.argmax(1))
        save_outputs(
            setting,
            name,
            pv,
            pt,
            {"params": best, "val_f1_search": val_f1, "test": m, "dims": X["train"].shape[1],
             "seconds": time.time() - t},
        )
        print(f"  [{setting}] {name:11s} val-F1 {val_f1:.3f}  test acc {m['acc']:.3f} F1 {m['f1']:.3f}  "
              f"{best}  ({time.time() - t:.0f}s)", flush=True)


def main(settings) -> None:
    seed_everything()
    all_paths = sorted({p for st in settings for k in ("train", "val", "test") for p, _ in load_split(st)[k]})
    feats = extract_all(all_paths)
    for st in settings:
        run_setting(st, feats)


if __name__ == "__main__":
    main(sys.argv[1:] or SETTINGS)
