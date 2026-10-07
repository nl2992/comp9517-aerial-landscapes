"""Two-stage transfer learning for ResNet-18 and EfficientNet-B3 (ImageNet weights, torchvision).

Stage 1  linear head on frozen, cached backbone features; Optuna (TPE, 20 trials) picks
         lr and weight decay by validation macro-F1.
Stage 2  fine-tuning from that head (linear probe, then fine-tune): ResNet-18 all layers,
         EfficientNet-B3 the last MBConv stage + head conv + classifier. Adam with head
         lr = min(lr*, 1e-3) and backbone lr = 1e-4, weight decay wd*, cosine schedule, at most
         15 epochs, early stopping (patience 4) on validation macro-F1, best epoch restored.

    python code/experiments/cnn.py <resnet18|effnet_b3> <setting> [--class-weighted]
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import models

from common import NUM_CLASSES, ROOT, RUNS, SEED, load_split, macro_metrics, save_outputs, seed_everything

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
BATCH = 64
MAX_EPOCHS, PATIENCE = 15, 4
HEAD_TRIALS, HEAD_EPOCHS = 20, 15
FT_HEAD_LR_MAX, FT_BACKBONE_LR = 1e-3, 1e-4

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_images(items) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty((len(items), 3, SIZE, SIZE), dtype=torch.uint8)
    for i, (p, _) in enumerate(items):
        img = Image.open(ROOT / p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        X[i] = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
    return X, torch.tensor([y for _, y in items])


def prep(xb: torch.Tensor, train: bool) -> torch.Tensor:
    x = xb.to(DEV, non_blocking=True).float().div_(255)
    if train:  # per-sample horizontal flip and rotation in [-15, 15] degrees
        n = x.shape[0]
        flip = torch.rand(n, device=DEV) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        a = (torch.rand(n, device=DEV) * 30 - 15) * math.pi / 180
        theta = torch.zeros(n, 2, 3, device=DEV)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    return (x - MEAN) / STD


class Net(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
            self.dim, self.head = 512, nn.Linear(512, NUM_CLASSES)
        else:
            m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten())
            self.dim = 1536
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, NUM_CLASSES))

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable_backbone(self) -> list[nn.Module]:
        if self.arch == "resnet18":
            return [self.backbone]
        feats = self.backbone[0]
        return [feats[7], feats[8]]  # last MBConv stage + 1x1 head conv

    def set_mode(self, train: bool) -> None:
        self.train(train)
        if train and self.arch != "resnet18":  # keep frozen BatchNorm statistics fixed
            self.backbone.eval()
            for m in self.trainable_backbone():
                m.train()


@torch.no_grad()
def embed(backbone: nn.Module, X: torch.Tensor) -> torch.Tensor:
    backbone.eval()
    return torch.cat([backbone(prep(X[i : i + 128], False)).cpu() for i in range(0, len(X), 128)])


@torch.no_grad()
def predict(net: Net, X: torch.Tensor) -> np.ndarray:
    net.eval()
    out = [F.softmax(net(prep(X[i : i + 128], False)), 1).cpu() for i in range(0, len(X), 128)]
    return torch.cat(out).numpy()


def train_head(head, Ftr, ytr, lr, wd, weight):
    head = head.to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(HEAD_EPOCHS):
        head.train()
        for idx in torch.randperm(len(Ftr), generator=g).split(256):
            loss = F.cross_entropy(head(Ftr[idx].to(DEV)), ytr[idx].to(DEV), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def head_f1(head, Fva, yva) -> float:
    head.eval()
    with torch.no_grad():
        pred = head(Fva.to(DEV)).argmax(1).cpu()
    return f1_score(yva, pred, average="macro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=["resnet18", "effnet_b3"])
    ap.add_argument("setting")
    ap.add_argument("--class-weighted", action="store_true", help="inverse-frequency weighted cross-entropy")
    a = ap.parse_args()
    name = a.arch + ("_cw" if a.class_weighted else "")
    seed_everything()
    t0 = time.time()

    s = load_split(a.setting)
    Xtr, ytr = load_images(s["train"])
    Xva, yva = load_images(s["val"])
    Xte, yte = load_images(s["test"])
    counts = torch.bincount(ytr, minlength=NUM_CLASSES).float()
    weight = (counts.sum() / (NUM_CLASSES * counts)).to(DEV) if a.class_weighted else None
    print(f"[{a.setting}/{name}] loaded {len(Xtr)}/{len(Xva)}/{len(Xte)} in {time.time() - t0:.0f}s", flush=True)

    # Stage 1: head search on cached features
    net = Net(a.arch).to(DEV)
    Ftr, Fva = embed(net.backbone, Xtr), embed(net.backbone, Xva)

    def objective(trial):
        torch.manual_seed(SEED)
        head = copy.deepcopy(net.head)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 5e-3, log=True)
        return head_f1(train_head(head, Ftr, ytr, lr, wd, weight), Fva, yva)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=HEAD_TRIALS)
    lr, wd = study.best_params["lr"], study.best_params["wd"]
    torch.manual_seed(SEED)
    net.head = train_head(net.head, Ftr, ytr, lr, wd, weight)
    print(f"  head search: lr={lr:.2e} wd={wd:.2e} val-F1={study.best_value:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # Stage 2: fine-tuning with early stopping on validation macro-F1
    for p in net.backbone.parameters():
        p.requires_grad = False
    bb_params = [p for m in net.trainable_backbone() for p in m.parameters()]
    for p in bb_params:
        p.requires_grad = True
    opt = torch.optim.Adam(
        [{"params": net.head.parameters(), "lr": min(lr, FT_HEAD_LR_MAX)}, {"params": bb_params, "lr": FT_BACKBONE_LR}],
        weight_decay=wd,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best_f1, best_state, best_ep, bad, curve = -1.0, None, 0, 0, []
    g = torch.Generator().manual_seed(SEED)
    for ep in range(1, MAX_EPOCHS + 1):
        te = time.time()
        net.set_mode(True)
        for idx in torch.randperm(len(Xtr), generator=g).split(BATCH):
            loss = F.cross_entropy(net(prep(Xtr[idx], True)), ytr[idx].to(DEV), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        f1 = f1_score(yva, predict(net, Xva).argmax(1), average="macro")
        curve.append(f1)
        print(f"  ep {ep:02d} val-F1 {f1:.4f} ({time.time() - te:.0f}s)", flush=True)
        if f1 > best_f1 + 1e-4:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)

    pv, pt = predict(net, Xva), predict(net, Xte)
    m = macro_metrics(yte.numpy(), pt.argmax(1))
    out = RUNS / a.setting / name
    save_outputs(
        a.setting,
        name,
        pv,
        pt,
        {"head_lr": lr, "head_wd": wd, "head_val_f1": study.best_value, "best_epoch": best_ep,
         "val_f1_curve": curve, "val_f1": best_f1, "test": m, "seconds": time.time() - t0,
         "class_weighted": a.class_weighted},
    )
    torch.save(net.state_dict(), out / "model.pt")
    print(f"[{a.setting}/{name}] best ep {best_ep} val-F1 {best_f1:.4f} | test acc {m['acc']:.4f} "
          f"F1 {m['f1']:.4f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
