"""Build a side-by-side comparison table from both models' eval reports.

Run after: python src/eval.py --model unet  AND  --model deeplabv3plus
Usage:     python src/compare.py
"""
import json

import config as C
from models import build_model, count_params

MODELS = ["unet", "deeplabv3plus"]


def _best_params(model):
    p = C.RESULTS_DIR / f"{model}_best_params.json"
    if p.exists():
        return json.loads(p.read_text()).get("params", {})
    return {}


def main():
    rows = []
    for m in MODELS:
        rep = json.loads((C.RESULTS_DIR / f"{m}_report.json").read_text())
        bp = _best_params(m)
        encoder = bp.get("encoder", C.ENCODER)
        params = count_params(build_model(m, encoder=encoder)) / 1e6
        rows.append((m, rep, params, bp))

    # Markdown table: per-class Dice + mean FG Dice/IoU + params.
    headers = ["Model"] + [f"Dice {c}" for c in C.CLASS_NAMES[1:]] + \
              ["mean FG Dice", "mean FG IoU", "Params (M)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for m, rep, params, _bp in rows:
        cells = [m]
        for c in C.CLASS_NAMES[1:]:
            cells.append(f"{rep['per_class'][c]['dice']:.3f}")
        cells += [f"{rep['mean_fg_dice']:.3f}", f"{rep['mean_fg_iou']:.3f}",
                  f"{params:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")

    # Second table: best hyperparameters found by Optuna.
    bp_lines = ["", "## Best hyperparameters (Optuna)", "",
                "| Model | lr | batch_size | encoder | aug_strength |",
                "|---|---|---|---|---|"]
    for m, _rep, _params, bp in rows:
        if bp:
            bp_lines.append(
                f"| {m} | {bp.get('lr', '')} | {bp.get('batch_size', '')} | "
                f"{bp.get('encoder', '')} | {bp.get('aug_strength', '')} |")

    table = "\n".join(lines)
    out = C.RESULTS_DIR / "comparison.md"
    out.write_text("# U-Net vs DeepLabV3+ — Test Results\n\n" + table + "\n"
                   + "\n".join(bp_lines) + "\n")
    print(table)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
