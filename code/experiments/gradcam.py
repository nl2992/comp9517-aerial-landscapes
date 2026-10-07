"""Grad-CAM for the most frequent confusion of each CNN in a given setting.

    python code/experiments/gradcam.py [setting]
Writes paper/figs/gradcam_<setting>.pdf and runs/<setting>/gradcam.json.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from analyze import FIGS, INK
from cnn import DEV, SIZE, Net, prep
from common import CLASSES, NUM_CLASSES, ROOT, RUNS, load_split

N_WRONG = 3


def cam_layer(net: Net):
    return net.backbone[7] if net.arch == "resnet18" else net.backbone[0][8]


def grad_cam(net: Net, x: torch.Tensor, cls: int) -> np.ndarray:
    store = {}
    h = cam_layer(net).register_forward_hook(lambda m, i, o: store.update(a=o))
    net.eval()
    x = x.clone().requires_grad_(True)
    out = net(x)
    h.remove()
    a = store["a"]
    g = torch.autograd.grad(out[0, cls], a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def main(setting: str) -> None:
    s = load_split(setting)
    y = np.array([c for _, c in s["test"]])
    paths = [p for p, _ in s["test"]]
    nets, preds = {}, {}
    for arch in ("resnet18", "effnet_b3"):
        net = Net(arch).to(DEV)
        net.load_state_dict(torch.load(RUNS / setting / arch / "model.pt", map_location=DEV))
        nets[arch] = net
        preds[arch] = np.load(RUNS / setting / arch / "test_prob.npy").argmax(1)

    # Most frequent off-diagonal confusion of ResNet-18; show it for both networks.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(cm, (y, preds["resnet18"]), 1)
    np.fill_diagonal(cm, 0)
    t, p = np.unravel_index(cm.argmax(), cm.shape)
    wrong = np.where((y == t) & (preds["resnet18"] == p))[0][:N_WRONG]
    right = np.where((y == t) & (preds["resnet18"] == t) & (preds["effnet_b3"] == t))[0][:1]
    idx = list(right) + list(wrong)
    info = {"true": CLASSES[t], "pred": CLASSES[p], "count": int(cm[t, p]),
            "b3_same_confusion": int(((y == t) & (preds["effnet_b3"] == p)).sum()),
            "examples": [{"path": paths[i], "r18": CLASSES[preds["resnet18"][i]], "b3": CLASSES[preds["effnet_b3"][i]]}
                         for i in idx]}

    fig, axes = plt.subplots(3, len(idx), figsize=(1.15 * len(idx), 3.55))
    for j, i in enumerate(idx):
        img = np.asarray(Image.open(ROOT / paths[i]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        x = prep(torch.from_numpy(img.copy()).permute(2, 0, 1)[None], False)
        axes[0, j].imshow(img)
        axes[0, j].set_title("correct" if j == 0 else f"error {j}", fontsize=7, color=INK)
        for r, arch in ((1, "resnet18"), (2, "effnet_b3")):
            pc = int(preds[arch][i])
            axes[r, j].imshow(img)
            axes[r, j].imshow(grad_cam(nets[arch], x, pc), cmap="jet", alpha=0.45)
            axes[r, j].set_xlabel(f"$\\rightarrow$ {CLASSES[pc]}", fontsize=6.5, labelpad=1, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, 0].set_ylabel("image", fontsize=7)
    axes[1, 0].set_ylabel("ResNet-18", fontsize=7)
    axes[2, 0].set_ylabel("EffNet-B3", fontsize=7)
    fig.tight_layout(pad=0.2, h_pad=0.6, w_pad=0.2)
    fig.savefig(FIGS / f"gradcam_{setting}.pdf", dpi=220)
    json.dump(info, open(RUNS / setting / "gradcam.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "longtail")
