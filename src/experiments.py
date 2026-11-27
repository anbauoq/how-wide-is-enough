"""Compare widths and learning rates using training and validation data only."""

from pathlib import Path

import pandas as pd

from .evaluate import evaluate_model
from .train import train_model


def run_experiments(x_train, y_train, x_val, y_val, widths=(4, 16, 64),
                    learning_rates=(0.01,), seeds=(21,), activation="sigmoid",
                    epochs=1500, patience=100, min_delta=1e-6, csv_path=None,
                    domain=None, reduce_lr=False, lr_patience=30, lr_factor=0.5):
    """Compare widths, learning rates, and seeds on the same train/validation data.

    Accepts (n,) or (n, 1) arrays; training options go to train_model.
    Returns (results, trained_runs): a DataFrame of settings and errors,
    and a dictionary mapping (width, learning_rate, seed) to (model, history).
    Models already have their best validation weights restored.

    Select settings by mean validation_mse across seeds. Test data stay out
    of the search. csv_path saves only the table; models stay in memory.
    domain only labels the supplied data for plots. learning_rate is the
    initial rate; final_learning_rate is the rate used in the last epoch.
    """
    widths, learning_rates, seeds = tuple(widths), tuple(learning_rates), tuple(seeds)
    if x_val is None or y_val is None:
        raise ValueError("tuning requires validation data")
    if not widths or not learning_rates or not seeds:
        raise ValueError("widths, learning_rates, and seeds must be nonempty")
    label = "Experiment" if domain is None else f"[{domain[0]:g}, {domain[1]:g}]"
    rows = []
    trained_runs = {}
    for width in widths:
        for rate in learning_rates:
            for seed in seeds:
                model, history = train_model(
                    x_train, y_train, x_val, y_val, width=width,
                    activation=activation, epochs=epochs, learning_rate=rate,
                    seed=seed, patience=patience, min_delta=min_delta,
                    reduce_lr=reduce_lr, lr_patience=lr_patience, lr_factor=lr_factor,
                )
                row = {
                    "domain": label,
                    "width": width, "learning_rate": rate, "seed": seed,
                    "activation": activation, "optimizer": "adam",
                    "epochs": epochs, "patience": patience, "min_delta": min_delta,
                    "reduce_lr": reduce_lr,
                    "lr_patience": lr_patience, "lr_factor": lr_factor,
                    "final_learning_rate": history["learning_rate"][-1],
                    "epochs_run": history["epochs_run"],
                    "best_epoch": history["best_epoch"],
                    "stopped_early": history["stopped_early"],
                    "training_mse": evaluate_model(model, x_train, y_train)["mse"],
                    "validation_mse": evaluate_model(model, x_val, y_val)["mse"],
                }
                rows.append(row)
                trained_runs[(width, rate, seed)] = (model, history)
    results = pd.DataFrame(rows)
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)
    return results, trained_runs


data = prepare_data(affine)