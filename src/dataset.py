"""NIfTI loading, HU windowing, axial slicing, patient-level split, and Dataset."""
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

import config as C


def list_volumes():
    """Return sorted (image_path, mask_path) pairs, matched by filename stem."""
    imgs = sorted(Path(C.IMAGES_DIR).glob("*.nii.gz"))
    masks_dir = Path(C.MASKS_DIR)
    pairs = []
    for img in imgs:
        mask = masks_dir / img.name
        if not mask.exists():
            # fall back: match by stem (mask filenames sometimes differ slightly)
            cands = list(masks_dir.glob(img.name.split(".")[0] + "*"))
            mask = cands[0] if cands else None
        if mask is not None:
            pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(
            f"No matched image/mask pairs. Checked {C.IMAGES_DIR} and {C.MASKS_DIR}."
        )
    return pairs


def split_volumes(pairs):
    """Patient-level split -> (train, val, test) lists of (img, mask) pairs."""
    idx = list(range(len(pairs)))
    random.Random(C.SPLIT_SEED).shuffle(idx)
    test_idx = set(idx[: C.N_TEST])
    val_idx = set(idx[C.N_TEST : C.N_TEST + C.N_VAL])
    train, val, test = [], [], []
    for i, p in enumerate(pairs):
        (test if i in test_idx else val if i in val_idx else train).append(p)
    return train, val, test


def _window_normalize(slice_hu):
    """Clip to lung HU window and min-max normalize to [0,1]."""
    lo, hi = C.HU_WINDOW
    s = np.clip(slice_hu, lo, hi)
    return (s - lo) / (hi - lo)


def _remap_mask(mask):
    """Ensure mask labels are in {0..NUM_CLASSES-1}. Dataset already uses 0..3."""
    m = mask.astype(np.int64)
    m[m >= C.NUM_CLASSES] = 0
    m[m < 0] = 0
    return m


def extract_slices(pairs):
    """Load volumes and return list of (image_2d float32, mask_2d int64) axial slices.

    Slices with too few foreground (non-background) pixels are dropped.
    """
    samples = []
    for img_path, mask_path in pairs:
        vol = nib.load(str(img_path)).get_fdata()          # H, W, D (HU)
        msk = nib.load(str(mask_path)).get_fdata()
        vol = _window_normalize(vol)
        msk = _remap_mask(msk)
        for z in range(vol.shape[2]):
            m = msk[:, :, z]
            if (m > 0).sum() < C.MIN_FG_PIXELS:
                continue
            img = resize(vol[:, :, z], (C.IMG_SIZE, C.IMG_SIZE),
                         order=1, preserve_range=True, anti_aliasing=True)
            mr = resize(m, (C.IMG_SIZE, C.IMG_SIZE),
                        order=0, preserve_range=True, anti_aliasing=False)
            samples.append((img.astype(np.float32), mr.astype(np.int64)))
    return samples


class CTSegDataset(Dataset):
    """2D axial CT slice dataset. Returns (image[3,H,W] float, mask[H,W] long)."""

    def __init__(self, samples, augment=None):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, mask = self.samples[i]
        if self.augment is not None:
            out = self.augment(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        img = np.repeat(img[None], 3, axis=0)              # 1->3 channels (ImageNet)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def build_augment(strength="light"):
    """Train-time augmentation by strength: 'none' | 'light' | 'strong'.

    Returns None if albumentations is missing or strength == 'none'.
    """
    if strength == "none":
        return None
    try:
        import albumentations as A
    except ImportError:
        return None
    tfms = [A.HorizontalFlip(p=0.5)]
    if strength == "strong":
        tfms += [
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        ]
    return A.Compose(tfms)
