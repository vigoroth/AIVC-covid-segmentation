"""Segmentation metrics: per-class Dice, IoU, and pixel confusion matrix."""
import numpy as np
import torch


@torch.no_grad()
def confusion_matrix(pred, target, num_classes):
    """Accumulate a num_classes x num_classes pixel confusion matrix (rows=GT)."""
    k = (target >= 0) & (target < num_classes)
    idx = num_classes * target[k].astype(np.int64) + pred[k].astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def dice_iou_from_cm(cm):
    """Per-class Dice and IoU from a confusion matrix. Returns (dice[], iou[])."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp          # predicted class but not GT (columns)
    fn = cm.sum(1) - tp          # GT class but not predicted (rows)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-8)
    iou = tp / np.maximum(tp + fp + fn, 1e-8)
    return dice, iou


def sensitivity_specificity_from_cm(cm):
    """Per-class sensitivity (recall) and specificity from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - (tp + fp + fn)
    sens = tp / np.maximum(tp + fn, 1e-8)
    spec = tn / np.maximum(tn + fp, 1e-8)
    return sens, spec
