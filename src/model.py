"""Prediction and BFGS training for the one-hidden-layer sigmoid network."""

import numpy as np
from scipy.optimize import minimize


def predict(parameters, x):
    """Evaluate a network whose parameters are ``(a, b, c)`` concatenated."""
    # Each neuron has an output weight a, input weight b, and bias c.
    a, b, c = np.split(parameters, 3)
    z = np.clip(np.asarray(x)[:, None] * b + c, -700, 700)
    return (1 / (1 + np.exp(-z))) @ a


def _train_once(x_train, y_train, x_validation, y_validation, width, seed,
                max_iterations, patience, min_delta):
    rng = np.random.default_rng(seed)
    initial = rng.normal(0, 0.5, 3 * width)
    best_parameters = initial.copy()
    best_validation_error = float(np.max(np.abs(
        predict(initial, x_validation) - y_validation
    )))
    patience_reference = best_validation_error
    waiting = 0
    n = len(x_train)

    def loss_and_gradient(parameters):
        # BFGS minimizes training MSE; this also gives it the gradient it needs.
        a, b, c = np.split(parameters, 3)
        z = np.clip(x_train[:, None] * b + c, -700, 700)
        hidden = 1 / (1 + np.exp(-z))
        residual = hidden @ a - y_train
        loss = np.mean(residual**2)
        slope = (2 * residual / n)[:, None] * a * hidden * (1 - hidden)
        gradient = np.concatenate((
            2 * hidden.T @ residual / n,
            np.sum(slope * x_train[:, None], axis=0),
            np.sum(slope, axis=0),
        ))
        return loss, gradient

    def stop_on_plateau(parameters):
        nonlocal best_parameters, best_validation_error, patience_reference, waiting
        error = float(np.max(np.abs(
            predict(parameters, x_validation) - y_validation
        )))
        # Keep the best checkpoint, and stop if validation error stalls.
        if error < best_validation_error:
            best_parameters = parameters.copy()
            best_validation_error = error
        if error < patience_reference - min_delta:
            patience_reference = error
            waiting = 0
        else:
            waiting += 1
        if waiting >= patience:
            raise StopIteration

    try:
        minimize(
            loss_and_gradient, initial, method="BFGS", jac=True,
            callback=stop_on_plateau,
            options={"maxiter": max_iterations, "gtol": 1e-8},
        )
    except StopIteration:
        pass

    gradient_norm = float(np.linalg.norm(loss_and_gradient(best_parameters)[1]))
    return {
        "parameters": best_parameters,
        "validation_max_error": best_validation_error,
        "gradient_norm": gradient_norm,
        "seed": seed,
    }


def train_model(x_train, y_train, x_validation, y_validation, width,
                seeds=range(21, 31), max_iterations=1500, patience=100,
                min_delta=1e-8):
    """Train random starts and return the one with the smallest gradient norm."""
    # Different seeds give different starting weights; keep the run with the
    # smallest gradient norm at its saved checkpoint.
    fits = [
        _train_once(
            x_train, y_train, x_validation, y_validation, width, seed,
            max_iterations, patience, min_delta,
        )
        for seed in seeds
    ]
    if not fits:
        raise ValueError("seeds must contain at least one seed")
    return min(fits, key=lambda fit: fit["gradient_norm"])
