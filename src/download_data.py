"""Download + unzip the COVID-19 CT Lung and Infection Segmentation Dataset.

Zenodo record 3757476 (CC-BY-4.0). Fetches the image volumes and the combined
lung+infection masks (~1.1 GB total).

Usage:  python src/download_data.py
"""
import urllib.request
import zipfile

import config as C

BASE = "https://zenodo.org/records/3757476/files"
# Both zips contain identically-named .nii.gz files, so each MUST extract into its
# own subfolder (matching IMAGES_DIR / MASKS_DIR in config) or masks overwrite images.
FILES = {
    "COVID-19-CT-Seg_20cases.zip": (
        f"{BASE}/COVID-19-CT-Seg_20cases.zip?download=1", C.IMAGES_DIR),
    "Lung_and_Infection_Mask.zip": (
        f"{BASE}/Lung_and_Infection_Mask.zip?download=1", C.MASKS_DIR),
}


def _progress(block, bsize, total):
    done = block * bsize
    pct = 100 * done / total if total > 0 else 0
    print(f"\r  {done/1e6:7.1f} MB / {total/1e6:7.1f} MB ({pct:5.1f}%)", end="")


def main():
    C.DATA_DIR.mkdir(exist_ok=True)
    for name, (url, dest) in FILES.items():
        zip_path = C.DATA_DIR / name
        if not zip_path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, zip_path, _progress)
            print()
        else:
            print(f"{name} already present, skipping download.")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Unzipping {name} -> {dest} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    print("\nDone. Verify these dirs exist:")
    print(f"  images: {C.IMAGES_DIR}")
    print(f"  masks : {C.MASKS_DIR}")
    print("If folder names differ, update IMAGES_DIR/MASKS_DIR in src/config.py.")


if __name__ == "__main__":
    main()
