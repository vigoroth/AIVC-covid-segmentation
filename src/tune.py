"""Optuna hyperparameter search per model. Records every trial + study artifacts.

Usage:
    python src/tune.py --model unet
    python src/tune.py --model deeplabv3plus
    python src/tune.py --model unet --trials 1 --epochs 2   # quick dry run

Outputs (all in results/ unless noted):
    <model>_trial<N>_history.json / _curves.png   per-trial per-epoch metrics
    checkpoints/<model>_trial<N>.pth              per-trial best weights
    checkpoints/<model>_best.pth                  best trial overall (eval-ready)
    <model>_best_params.json                      winning hyperparameters
    <model>_trials.csv                            all trials (params + value)
    optuna_<model>.db                             SQLite study (resumable)
    optuna_<model>_{history,importance,parallel,slice}.png   study plots
"""
import argparse
import json
import shutil

import optuna
from optuna.visualization import matplotlib as ovm

import config as C
from train import train_model

SEARCH = dict(
    lr=(1e-5, 1e-2),
    batch_size=[4, 8, 16],
    encoder=["resnet18", "resnet34", "efficientnet-b0"],
    aug_strength=["none", "light", "strong"],
)


def make_objective(model_name, epochs):
    def objective(trial):
        params = dict(
            lr=trial.suggest_float("lr", *SEARCH["lr"], log=True),
            batch_size=trial.suggest_categorical("batch_size", SEARCH["batch_size"]),
            encoder=trial.suggest_categorical("encoder", SEARCH["encoder"]),
            aug_strength=trial.suggest_categorical("aug_strength", SEARCH["aug_strength"]),
        )
        tag = f"{model_name}_trial{trial.number}"
        print(f"\n=== {tag}  params={params} ===")
        _, best_val_dice, ckpt = train_model(model_name, params, epochs, tag, trial=trial)
        trial.set_user_attr("ckpt", str(ckpt))
        return best_val_dice
    return objective


def save_study_plots(study, model_name):
    plots = {
        "history": ovm.plot_optimization_history,
        "importance": ovm.plot_param_importances,
        "parallel": ovm.plot_parallel_coordinate,
        "slice": ovm.plot_slice,
    }
    for name, fn in plots.items():
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            out = C.RESULTS_DIR / f"optuna_{model_name}_{name}.png"
            fig.tight_layout(); fig.savefig(out, dpi=120)
            print(f"Saved {out}")
        except Exception as e:        # importance needs >1 completed trial, etc.
            print(f"[warn] could not render {name} plot: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "deeplabv3plus"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()

    storage = f"sqlite:///{C.RESULTS_DIR / f'optuna_{args.model}.db'}"
    study = optuna.create_study(
        study_name=args.model, direction="maximize", storage=storage,
        load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(make_objective(args.model, args.epochs), n_trials=args.trials)

    best = study.best_trial
    print(f"\nBest trial #{best.number}: val_dice={best.value:.4f} params={best.params}")

    # Promote best trial's checkpoint to the eval-ready name.
    best_ckpt = best.user_attrs.get("ckpt", str(C.CKPT_DIR / f"{args.model}_trial{best.number}.pth"))
    shutil.copy(best_ckpt, C.CKPT_DIR / f"{args.model}_best.pth")

    (C.RESULTS_DIR / f"{args.model}_best_params.json").write_text(
        json.dumps(dict(value=best.value, params=best.params,
                        trial=best.number), indent=2))
    study.trials_dataframe().to_csv(
        C.RESULTS_DIR / f"{args.model}_trials.csv", index=False)
    save_study_plots(study, args.model)
    print(f"Done. best ckpt -> {C.CKPT_DIR / f'{args.model}_best.pth'}")


if __name__ == "__main__":
    main()
