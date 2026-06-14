"""Evaluate a trained model on the test split: metrics, confusion matrix, curves, overlays.

Usage:
    python src/eval.py --model unet
    python src/eval.py --model deeplabv3plus
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from dataset import CTSegDataset, extract_slices, list_volumes, split_volumes
from metrics import (confusion_matrix, dice_iou_from_cm,
                     sensitivity_specificity_from_cm)
from models import build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    for img, mask in loader:
        pred = model(img.to(device)).argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, mask.numpy(), C.NUM_CLASSES)
    return cm


def _best_params(model_name):
    """Best Optuna params if present, else None (plain-train fallback)."""
    p = C.RESULTS_DIR / f"{model_name}_best_params.json"
    return json.loads(p.read_text()) if p.exists() else None


def _history_path(model_name):
    """Best-trial history if tuned, else the plain-train history file."""
    bp = _best_params(model_name)
    if bp is not None:
        cand = C.RESULTS_DIR / f"{model_name}_trial{bp['trial']}_history.json"
        if cand.exists():
            return cand
    return C.RESULTS_DIR / f"{model_name}_history.json"


def plot_curves(model_name):
    hist = json.loads(_history_path(model_name).read_text())
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set(title=f"{model_name} loss", xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_dice"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_dice"] for h in hist], label="val")
    ax[1].set(title=f"{model_name} mean FG Dice", xlabel="epoch", ylabel="dice"); ax[1].legend()
    fig.tight_layout()
    out = C.RESULTS_DIR / f"{model_name}_curves.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved {out}")


def plot_confusion(cm, model_name):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)   # row-normalized
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(C.NUM_CLASSES), yticks=range(C.NUM_CLASSES),
           xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES,
           xlabel="Predicted", ylabel="Ground truth",
           title=f"{model_name} confusion (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    out = C.RESULTS_DIR / f"{model_name}_confusion.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved {out}")


def save_overlays(model, samples, device, model_name, n=4):
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    model.eval()
    for r in range(n):
        img, mask = samples[r * max(1, len(samples) // n)]
        x = torch.from_numpy(np.repeat(img[None], 3, 0)).float()[None].to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy()
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].set_title("CT")
        ax[r, 1].imshow(mask, vmin=0, vmax=C.NUM_CLASSES - 1, cmap="viridis")
        ax[r, 1].set_title("GT")
        ax[r, 2].imshow(pred, vmin=0, vmax=C.NUM_CLASSES - 1, cmap="viridis")
        ax[r, 2].set_title("Pred")
        for c in range(3):
            ax[r, c].axis("off")
    fig.tight_layout()
    out = C.RESULTS_DIR / f"{model_name}_overlays.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "deeplabv3plus"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list_volumes()
    _, _, test_p = split_volumes(pairs)
    test_s = extract_slices(test_p)
    test_dl = DataLoader(CTSegDataset(test_s), C.BATCH_SIZE, shuffle=False,
                         num_workers=C.NUM_WORKERS)

    # Build with the encoder that produced the saved weights (else state mismatch).
    bp = _best_params(args.model)
    encoder = bp["params"]["encoder"] if bp else C.ENCODER
    model = build_model(args.model, encoder=encoder).to(device)
    model.load_state_dict(torch.load(C.CKPT_DIR / f"{args.model}_best.pth",
                                     map_location=device))

    cm = evaluate(model, test_dl, device)
    dice, iou = dice_iou_from_cm(cm)
    sens, spec = sensitivity_specificity_from_cm(cm)
    report = {
        "model": args.model,
        "per_class": {
            C.CLASS_NAMES[i]: dict(dice=float(dice[i]), iou=float(iou[i]),
                                   sensitivity=float(sens[i]),
                                   specificity=float(spec[i]))
            for i in range(C.NUM_CLASSES)
        },
        "mean_fg_dice": float(dice[1:].mean()),
        "mean_fg_iou": float(iou[1:].mean()),
        "confusion_matrix": cm.tolist(),
    }
    out = C.RESULTS_DIR / f"{args.model}_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["per_class"], indent=2))
    print(f"mean FG Dice={report['mean_fg_dice']:.4f}  saved -> {out}")

    plot_curves(args.model)
    plot_confusion(cm, args.model)
    save_overlays(model, test_s, device, args.model)


if __name__ == "__main__":
    main()
