"""Prediction and errors measured on the supplied evaluation points."""

import numpy as np
import torch


def predict(model, x):
    """Predict for x of shape (n, 1) or (n,); return a NumPy array (n, 1)."""
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(x, dtype=torch.float32)
        if inputs.ndim not in (1, 2) or (inputs.ndim == 2 and inputs.shape[1] != 1):
            raise ValueError("x must have shape (n,) or (n, 1)")
        inputs = inputs.reshape(-1, 1)
        return model(inputs).cpu().numpy()


def evaluate_model(model, x, y):
    """Compare predictions with y; x and y must contain the same n points.

    Arrays can have shape (n, 1) or (n,). Returns a dictionary with float
    values: mse, rmse, max_absolute_error. Maximum error is measured only
    on these points, not over an entire continuous interval.
    """
    predictions = predict(model, x)
    targets = np.asarray(y, dtype=float)
    if targets.ndim not in (1, 2) or (targets.ndim == 2 and targets.shape[1] != 1):
        raise ValueError("y must have shape (n,) or (n, 1)")
    targets = targets.reshape(-1, 1)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("x and y must have matching, nonempty shapes")
    errors = predictions - targets
    mse = float(np.mean(errors**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }
