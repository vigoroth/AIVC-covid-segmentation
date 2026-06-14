"""Training core. Importable `train_model()` + standalone CLI.

CLI (uses config defaults):
    python src/train.py --model unet
    python src/train.py --model deeplabv3plus
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from dataset import (CTSegDataset, build_augment, extract_slices,
                     list_volumes, split_volumes)
from metrics import confusion_matrix, dice_iou_from_cm
from models import build_model, count_params

# Slices are deterministic given the split, so cache them across trials/models.
_SLICE_CACHE = {}


def _get_slices():
    if not _SLICE_CACHE:
        pairs = list_volumes()
        train_p, val_p, test_p = split_volumes(pairs)
        _SLICE_CACHE["train"] = extract_slices(train_p)
        _SLICE_CACHE["val"] = extract_slices(val_p)
        print(f"Slices  train={len(_SLICE_CACHE['train'])} "
              f"val={len(_SLICE_CACHE['val'])}")
    return _SLICE_CACHE["train"], _SLICE_CACHE["val"]


def make_loss():
    dice = smp.losses.DiceLoss(mode="multiclass")
    ce = torch.nn.CrossEntropyLoss()
    return lambda logits, target: dice(logits, target) + ce(logits, target)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    for img, mask in tqdm(loader, leave=False):
        img, mask = img.to(device), mask.to(device)
        with torch.set_grad_enabled(train):
            logits = model(img)
            loss = loss_fn(logits, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * img.size(0)
        pred = logits.argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, mask.cpu().numpy(), C.NUM_CLASSES)
    dice, iou = dice_iou_from_cm(cm)
    return total_loss / len(loader.dataset), dice, iou


def _save_curves(history, tag):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(title=f"{tag} loss", xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_dice"] for h in history], label="train")
    ax[1].plot(ep, [h["val_dice"] for h in history], label="val")
    ax[1].set(title=f"{tag} mean FG Dice", xlabel="epoch", ylabel="dice"); ax[1].legend()
    fig.tight_layout()
    out = C.RESULTS_DIR / f"{tag}_curves.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def default_params():
    return dict(lr=C.LR, batch_size=C.BATCH_SIZE,
                encoder=C.ENCODER, aug_strength="strong")


def train_model(model_name, params, epochs, tag, trial=None, device=None):
    """Train one model with `params`; record per-epoch metrics. Returns
    (history, best_val_dice, best_ckpt_path).

    params keys: lr, batch_size, encoder, aug_strength.
    `tag` namespaces all output files. If `trial` (Optuna) given, reports
    intermediate val Dice for pruning.
    """
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_s, val_s = _get_slices()

    train_dl = DataLoader(
        CTSegDataset(train_s, augment=build_augment(params["aug_strength"])),
        batch_size=params["batch_size"], shuffle=True,
        num_workers=C.NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(CTSegDataset(val_s), batch_size=params["batch_size"],
                        shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=True)

    model = build_model(model_name, encoder=params["encoder"]).to(device)
    loss_fn = make_loss()
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=4)

    ckpt = C.CKPT_DIR / f"{tag}.pth"
    history, best, patience = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = run_epoch(model, train_dl, loss_fn, device, opt)
        va_loss, va_dice, va_iou = run_epoch(model, val_dl, loss_fn, device)
        mean_tr, mean_va = float(tr_dice[1:].mean()), float(va_dice[1:].mean())
        sched.step(mean_va)
        history.append(dict(
            epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
            train_dice=mean_tr, val_dice=mean_va,
            val_dice_per_class={C.CLASS_NAMES[i]: float(va_dice[i])
                                for i in range(C.NUM_CLASSES)},
            val_iou_per_class={C.CLASS_NAMES[i]: float(va_iou[i])
                               for i in range(C.NUM_CLASSES)}))
        print(f"[{tag}] E{epoch:02d} {time.time()-t0:.0f}s "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_dice={mean_va:.4f}")
        if mean_va > best:
            best, patience = mean_va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
        if trial is not None:
            trial.report(mean_va, epoch)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience >= C.EARLY_STOP_PATIENCE:
            print(f"[{tag}] early stop at epoch {epoch} (best={best:.4f})")
            break

    (C.RESULTS_DIR / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    _save_curves(history, tag)
    return history, best, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "deeplabv3plus"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    import shutil
    params = default_params()
    _, best, ckpt = train_model(args.model, params, args.epochs, tag=args.model)
    # eval.py expects <model>_best.pth
    eval_ckpt = C.CKPT_DIR / f"{args.model}_best.pth"
    shutil.copy(ckpt, eval_ckpt)
    print(f"Best val mean FG Dice={best:.4f}  ckpt={eval_ckpt}  params={params}")


if __name__ == "__main__":
    main()
