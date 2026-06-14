"""Pipeline sanity checks. Run after downloading data, before full training.

    python src/smoke_test.py
"""
import numpy as np

import config as C
from dataset import extract_slices, list_volumes, split_volumes


def main():
    pairs = list_volumes()
    assert len(pairs) == 20, f"Expected 20 volumes, got {len(pairs)}"
    print(f"OK: found {len(pairs)} image/mask pairs")

    train, val, test = split_volumes(pairs)
    train_set = {p[0].name for p in train}
    val_set = {p[0].name for p in val}
    test_set = {p[0].name for p in test}
    assert not (train_set & val_set) and not (train_set & test_set) \
        and not (val_set & test_set), "Patient-level split leak detected!"
    print(f"OK: no leakage  train={len(train)} val={len(val)} test={len(test)}")

    # Check one volume's slices and label range.
    s = extract_slices(test[:1])
    assert len(s) > 0, "No foreground slices extracted from a test volume"
    labels = np.unique(np.concatenate([m.ravel() for _, m in s]))
    assert labels.min() >= 0 and labels.max() < C.NUM_CLASSES, \
        f"Mask labels out of range: {labels}"
    img, mask = s[0]
    assert img.shape == (C.IMG_SIZE, C.IMG_SIZE) == mask.shape
    assert 0.0 <= img.min() and img.max() <= 1.0, "Image not normalized to [0,1]"
    print(f"OK: {len(s)} slices from 1 volume, labels={labels.tolist()}, "
          f"img range=[{img.min():.2f},{img.max():.2f}]")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
