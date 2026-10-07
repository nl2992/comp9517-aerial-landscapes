"""Turn saved probabilities into the paper's tables, figures and summary numbers.

    python code/experiments/analyze.py
Writes paper/tables/*.tex, paper/figs/*.pdf and runs/results.json.
"""

from __future__ import annotations

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, recall_score

from common import CLASSES, NUM_CLASSES, ROOT, RUNS, SEED, SETTINGS, load_split, macro_metrics

TABLES, FIGS = ROOT / "paper" / "tables", ROOT / "paper" / "figs"
BASE = ["lbp_knn", "colour_svm", "sift_svm", "stacking", "resnet18", "effnet_b3"]
LABEL = {
    "lbp_knn": "LBP + $k$NN", "colour_svm": "Colour hist.\\ + SVM", "sift_svm": "SIFT-BoW + SVM",
    "stacking": "Stacking (SVM, $k$NN, DT)", "resnet18": "ResNet-18", "effnet_b3": "EfficientNet-B3",
    "resnet18_cw": "ResNet-18 (weighted CE)", "effnet_b3_cw": "EfficientNet-B3 (weighted CE)",
}
SHORT = {"lbp_knn": "kNN", "colour_svm": "CSVM", "sift_svm": "BoW", "stacking": "Stack",
         "resnet18": "R18", "effnet_b3": "B3"}
ENSEMBLES = {
    "kNN + Stack": ["lbp_knn", "stacking"],
    "kNN + R18": ["lbp_knn", "resnet18"],
    "kNN + B3": ["lbp_knn", "effnet_b3"],
    "Stack + R18": ["stacking", "resnet18"],
    "Stack + B3": ["stacking", "effnet_b3"],
    "R18 + B3": ["resnet18", "effnet_b3"],
    "kNN + Stack + R18 + B3": ["lbp_knn", "stacking", "resnet18", "effnet_b3"],
    "All six": BASE,
}
SET_LABEL = {"balanced": "Balanced", "longtail": "Long-tailed", "rebalanced": "Rebalanced"}
# Reference categorical palette (dataviz skill), fixed order: blue, orange, aqua.
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_BOOT = 2000

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def load(setting, model):
    d = RUNS / setting / model
    return np.load(d / "val_prob.npy"), np.load(d / "test_prob.npy"), json.load(open(d / "meta.json"))


def hard_vote(probs):
    """Plurality vote; ties are broken by the summed probability of the tied classes.

    Breaking ties by voter order (e.g. Counter.most_common) makes any two-member ensemble
    return its first member's prediction whenever the members disagree.
    """
    votes = np.stack([p.argmax(1) for p in probs], 1)
    counts = np.stack([(votes == c).sum(1) for c in range(NUM_CLASSES)], 1)
    tied = counts == counts.max(1, keepdims=True)
    return np.where(tied, sum(probs), -np.inf).argmax(1)


def first_wins_vote(probs):
    """The original evaluation code's vote, kept only to document its behaviour."""
    from collections import Counter

    votes = np.stack([p.argmax(1) for p in probs], 1)
    return np.array([Counter(v).most_common(1)[0][0] for v in votes])


def soft_vote(probs):
    return np.mean(probs, 0).argmax(1)


def boot_ci(y, pred, rng):
    n = len(y)
    acc, f1 = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        m = macro_metrics(y[i], pred[i])
        acc.append(m["acc"])
        f1.append(m["f1"])
    return np.percentile(acc, [2.5, 97.5]).tolist(), np.percentile(f1, [2.5, 97.5]).tolist()


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n01, "b_only": n10, "p": p}


def q_stat(y, a, b):
    ca, cb = a == y, b == y
    n11, n00 = (ca & cb).sum(), (~ca & ~cb).sum()
    n10, n01 = (ca & ~cb).sum(), (~ca & cb).sum()
    return float((n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1))


def pct(x):
    return f"{100 * x:.1f}"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = {"settings": {}}
    preds, probs = {}, {}

    for st in SETTINGS:
        s = load_split(st)
        y = np.array([c for _, c in s["test"]])
        R["settings"][st] = {"train_counts": s["train_counts"], "base": {}, "ensembles": {}}
        models = BASE + ([m + "_cw" for m in ("resnet18", "effnet_b3")] if st == "longtail" else [])
        for m in models:
            if not (RUNS / st / m / "test_prob.npy").exists():
                continue
            _, pt, meta = load(st, m)
            probs[st, m], preds[st, m] = pt, pt.argmax(1)
            met = macro_metrics(y, preds[st, m])
            ci_acc, ci_f1 = boot_ci(y, preds[st, m], rng)
            rec = recall_score(y, preds[st, m], average=None, labels=range(NUM_CLASSES), zero_division=0)
            R["settings"][st]["base"][m] = {**met, "ci_acc": ci_acc, "ci_f1": ci_f1, "per_class_recall": rec.tolist(),
                                            "meta": meta}
        for name, members in ENSEMBLES.items():
            P = [probs[st, m] for m in members]
            r = {}
            for k, fn in (("hard", hard_vote), ("soft", soft_vote), ("first_wins", first_wins_vote)):
                r[k] = macro_metrics(y, fn(P))
            r["oracle_acc"] = float(np.mean(np.any([preds[st, m] == y for m in members], 0)))
            best = max(members, key=lambda m: R["settings"][st]["base"][m]["f1"])
            r["best_member"], r["best_member_f1"] = best, R["settings"][st]["base"][best]["f1"]
            r["mcnemar_soft_vs_best"] = mcnemar(y, soft_vote(P), preds[st, best])
            R["settings"][st]["ensembles"][name] = r
        main4 = ["lbp_knn", "stacking", "resnet18", "effnet_b3"]
        R["settings"][st]["q_stat"] = {f"{SHORT[a]}-{SHORT[b]}": q_stat(y, preds[st, a], preds[st, b])
                                       for a, b in itertools.combinations(main4, 2)}
        R["settings"][st]["mcnemar_b3_vs_r18"] = mcnemar(y, preds[st, "effnet_b3"], preds[st, "resnet18"])
        R["settings"][st]["y_test"] = y.tolist()

    # Paired tests across training settings (same test images)
    y = np.array(R["settings"]["balanced"]["y_test"])
    R["across_settings"] = {}
    for m in BASE + ["resnet18_cw", "effnet_b3_cw"]:
        r = {}
        for a, b in (("longtail", "balanced"), ("rebalanced", "longtail")):
            if (a, m) in preds and (b, m) in preds:
                r[f"{a}_vs_{b}"] = mcnemar(y, preds[a, m], preds[b, m])
        if m.endswith("_cw") and ("longtail", m) in preds:
            r["cw_vs_plain_longtail"] = mcnemar(y, preds["longtail", m], preds["longtail", m[:-3]])
        R["across_settings"][m] = r

    write_tables(R)
    figures(R, preds)
    json.dump(R, open(RUNS / "results.json", "w"), indent=1, default=float)
    print_summary(R)


def write_tables(R):
    S = R["settings"]
    # Table: base models across settings (accuracy, macro-F1 with 95% CI)
    rows = []
    for m in BASE:
        cells = []
        for st in SETTINGS:
            b = S[st]["base"][m]
            cells.append(f"{pct(b['acc'])} & {pct(b['f1'])} {{\\scriptsize[{pct(b['ci_f1'][0])}, {pct(b['ci_f1'][1])}]}}")
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
        if m == "stacking":
            rows.append(r"\midrule")
    for m in ("resnet18_cw", "effnet_b3_cw"):
        if m in S["longtail"]["base"]:
            b = S["longtail"]["base"][m]
            rows.append(f"{LABEL[m]} & -- & -- & {pct(b['acc'])} & {pct(b['f1'])} "
                        f"{{\\scriptsize[{pct(b['ci_f1'][0])}, {pct(b['ci_f1'][1])}]}} & -- & -- \\\\")
    (TABLES / "base.tex").write_text("\n".join(rows) + "\n")

    # Table: ensembles, macro-F1 for hard and soft voting, gain over best member
    rows = []
    for name in ENSEMBLES:
        cells = []
        for st in SETTINGS:
            e = S[st]["ensembles"][name]
            gain = e["soft"]["f1"] - e["best_member_f1"]
            star = "$^\\ast$" if e["mcnemar_soft_vs_best"]["p"] < 0.05 else ""
            cells.append(f"{pct(e['hard']['f1'])} & {pct(e['soft']['f1'])} & ${100 * gain:+.1f}${star}")
        rows.append(f"{name.replace('kNN', '$k$NN')} & " + " & ".join(cells) + r" \\")
    (TABLES / "ensembles.tex").write_text("\n".join(rows) + "\n")

    # Table: recall on head / middle / tail thirds of the long-tailed class ordering
    rows = []
    groups = {"Head": range(0, 5), "Middle": range(5, 10), "Tail": range(10, 15)}
    for m in ["stacking", "resnet18", "effnet_b3"]:
        cells = []
        for st in SETTINGS:
            rec = np.array(S[st]["base"][m]["per_class_recall"])
            cells += [pct(rec[list(g)].mean()) for g in groups.values()]
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    for m in ("resnet18_cw", "effnet_b3_cw"):
        if m in S["longtail"]["base"]:
            rec = np.array(S["longtail"]["base"][m]["per_class_recall"])
            rows.append(f"{LABEL[m]} & -- & -- & -- & " + " & ".join(pct(rec[list(g)].mean()) for g in groups.values())
                        + r" & -- & -- & -- \\")
    (TABLES / "groups.tex").write_text("\n".join(rows) + "\n")


def figures(R, preds):
    S = R["settings"]

    # Fig: macro-F1 of each base model under the three training settings (grouped bars)
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    x, w = np.arange(len(BASE)), 0.26
    for k, st in enumerate(SETTINGS):
        v = [S[st]["base"][m]["f1"] for m in BASE]
        lo = [S[st]["base"][m]["f1"] - S[st]["base"][m]["ci_f1"][0] for m in BASE]
        hi = [S[st]["base"][m]["ci_f1"][1] - S[st]["base"][m]["f1"] for m in BASE]
        ax.bar(x + (k - 1) * w, v, w - 0.03, color=C[k], label=SET_LABEL[st], zorder=2)
        ax.errorbar(x + (k - 1) * w, v, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.2, zorder=3)
    ax.set_xticks(x, [SHORT[m].replace("CSVM", "Colour").replace("BoW", "SIFT") for m in BASE])
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), handlelength=1, columnspacing=0.8, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "base_f1.pdf")
    plt.close(fig)

    # Fig: per-class recall vs. long-tail class index, long-tailed vs. rebalanced (small multiples)
    counts = S["longtail"]["train_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.9), sharey=True)
    for ax, st in zip(axes, ("longtail", "rebalanced")):
        for k, (m, mk) in enumerate((("stacking", "s"), ("resnet18", "o"), ("effnet_b3", "^"))):
            rec = S[st]["base"][m]["per_class_recall"]
            ax.plot(range(NUM_CLASSES), rec, color=C[k], lw=1.2, marker=mk, ms=3.2, label=LABEL[m].split(" (")[0])
        ax.set_title(f"{SET_LABEL[st]} training set", fontsize=8, color=INK)
        ax.set_xticks(range(NUM_CLASSES), [f"{c}\n{n}" for c, n in zip(CLASSES, counts)], rotation=90, fontsize=5.5)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Test recall")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.text(0.5, 0.005, "Class (number of real training images in the long-tailed set)", ha="center", fontsize=7,
             color=INK2)
    fig.tight_layout(pad=0.3, rect=(0.01, 0.04, 1, 1))
    fig.savefig(FIGS / "per_class_recall.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Fig: confusion matrices (row-normalised) for stacking and EfficientNet-B3, balanced setting
    y = np.array(S["balanced"]["y_test"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    for ax, m in zip(axes, ("stacking", "effnet_b3")):
        cm = confusion_matrix(y, preds["balanced", m], labels=range(NUM_CLASSES), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=90, fontsize=6)
        ax.set_yticks(range(NUM_CLASSES), CLASSES, fontsize=6)
        ax.set_title(f"{LABEL[m].split(' (')[0]} (balanced)", fontsize=8, color=INK)
        ax.set_xlabel("Predicted")
        for i, j in zip(*np.where(cm >= 0.05)):
            if i != j:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if cm[i, j] > 0.5 else INK)
    axes[0].set_ylabel("True")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01).ax.tick_params(labelsize=6)
    fig.savefig(FIGS / "confusion.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def print_summary(R):
    for st in SETTINGS:
        S = R["settings"][st]
        print(f"\n== {st}")
        for m, b in S["base"].items():
            print(f"  {m:13s} acc {b['acc']:.4f} F1 {b['f1']:.4f} CI {b['ci_f1'][0]:.3f}-{b['ci_f1'][1]:.3f}")
        for n, e in S["ensembles"].items():
            print(f"  {n:24s} hard {e['hard']['f1']:.4f} soft {e['soft']['f1']:.4f} first-wins {e['first_wins']['f1']:.4f} "
                  f"oracle {e['oracle_acc']:.4f} best {e['best_member']} {e['best_member_f1']:.4f} "
                  f"p={e['mcnemar_soft_vs_best']['p']:.3g}")
        print("  Q:", {k: round(v, 3) for k, v in S["q_stat"].items()}, " McNemar B3 vs R18:", S["mcnemar_b3_vs_r18"])
        for m in ("stacking", "resnet18", "effnet_b3", "resnet18_cw", "effnet_b3_cw"):
            if m in S["base"]:
                rec = np.array(S["base"][m]["per_class_recall"])
                print(f"  recall {m:13s} head {rec[:5].mean():.3f} mid {rec[5:10].mean():.3f} tail {rec[10:].mean():.3f}")
    print("\nAcross settings (a_only = right only under first setting):")
    for m, r in R["across_settings"].items():
        print(" ", m, {k: (v["a_only"], v["b_only"], round(v["p"], 4)) for k, v in r.items()})


if __name__ == "__main__":
    main()
