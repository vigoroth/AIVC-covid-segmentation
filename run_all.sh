#!/usr/bin/env bash
# ============================================================================
#  COVID-19 CT Segmentation - one-click pipeline (WSL/Linux + conda)
#  Creates env, installs deps, downloads data, trains + evaluates both models.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME=covidseg
PY_VER=3.10

# --- conda must be available ---
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found on PATH. Install Miniconda or run 'conda init bash'."
    exit 1
fi
# load conda's shell hook so 'conda activate' works inside a script
source "$(conda info --base)/etc/profile.d/conda.sh"

echo
echo "=== [1/7] Create conda env '$ENV_NAME' (python $PY_VER) ==="
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "Env '$ENV_NAME' already exists, skipping create."
else
    conda create -y -n "$ENV_NAME" python="$PY_VER"
fi

echo
echo "=== [2/7] Activate env ==="
conda activate "$ENV_NAME"

echo
echo "=== [3/7] Install PyTorch (CUDA 12.1) + project requirements ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo
echo "=== [4/7] Download dataset (~1.1 GB) ==="
python src/download_data.py

echo
echo "=== [5/7] Smoke test (pipeline sanity) ==="
python src/smoke_test.py

echo
echo "=== [6/7] Hyperparameter search (Optuna) for both models ==="
python src/tune.py --model unet
python src/tune.py --model deeplabv3plus

echo
echo "=== [7/7] Evaluate + compare ==="
python src/eval.py --model unet
python src/eval.py --model deeplabv3plus
python src/compare.py

echo
echo "============================================================"
echo " DONE. See results/ for curves, confusion matrices,"
echo " overlays, reports, and comparison.md"
echo "============================================================"
