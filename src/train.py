"""Shared Adam training with validation-based early stopping."""

from copy import deepcopy
import random

import numpy as np
import torch

from .models import ShallowNetwork


def set_seed(seed=21):
    """Seed Python, NumPy, and PyTorch; train_model calls this automatically.

    Data generation uses its own seed parameter. Reproducibility assumes
    the same software environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_arrays(x, y):
    """Check an input/target pair and return float32 column tensors."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    for name, values in (("x", x), ("y", y)):
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError(f"{name} must have shape (n,) or (n, 1)")
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if x.shape != y.shape or x.numel() == 0:
        raise ValueError("inputs and targets must have matching, nonempty shapes")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("data must be finite")
    return x, y


def train_model(x_train, y_train, x_val, y_val, width=32, activation="sigmoid",
                epochs=1500, learning_rate=0.01, seed=21,
                patience=100, min_delta=1e-6, reduce_lr=False,
                lr_patience=30, lr_factor=0.5):
    """Train with full-batch Adam on the CPU; return (model, history).

    Training and validation pairs must be finite, nonempty arrays of shape
    (n,) or (n, 1). Never use test data as validation.
    Stop after patience epochs without a min_delta improvement, up to epochs.
    Restore the weights with the lowest validation MSE.
    reduce_lr optionally multiplies the rate by lr_factor on a plateau;
    lr_patience must be shorter than early-stopping patience.

    history contains per-epoch training_mse, validation_mse, and learning_rate
    (the rate used for that update), plus best_epoch (1-based), epochs_run,
    and stopped_early. The restored weights may precede the final epoch.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be nonnegative and finite")
    if reduce_lr:
        if not isinstance(lr_patience, int) or not 1 <= lr_patience < patience:
            raise ValueError("lr_patience must be a positive integer smaller than patience")
        if not np.isfinite(lr_factor) or not 0 < lr_factor < 1:
            raise ValueError("lr_factor must be between 0 and 1")
    # Prepare the training and validation data.
    if x_val is None or y_val is None:
        raise ValueError("training requires both x_val and y_val")
    x_train_tensor, y_train_tensor = _prepare_arrays(x_train, y_train)
    x_val_tensor, y_val_tensor = _prepare_arrays(x_val, y_val)

    # Create and train the network.
    set_seed(seed)
    model = ShallowNetwork(width, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=min_delta, threshold_mode="abs",
        )
    mse = torch.nn.MSELoss()
    history = {"training_mse": [], "validation_mse": [], "learning_rate": []}
    best_loss = float("inf")  # Keep the actual lowest-MSE checkpoint.
    patience_reference = float("inf")  # Reset patience only for a sufficient improvement.
    waiting = 0
    model.train()
    for epoch in range(1, epochs + 1):
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        loss = mse(model(x_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss = mse(model(x_train_tensor), y_train_tensor).item()
        if not np.isfinite(train_loss):
            raise ValueError("Loss is not finite; try a lower learning rate")
        history["training_mse"].append(train_loss)
        with torch.no_grad():
            val_loss = mse(model(x_val_tensor), y_val_tensor).item()
        if not np.isfinite(val_loss):
            raise ValueError("Validation loss is not finite; try a lower learning rate")
        history["validation_mse"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        if val_loss < patience_reference - min_delta:
            patience_reference, waiting = val_loss, 0
        else:
            waiting += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        if waiting >= patience:
            break
    # Restore the model with the lowest validation MSE.
    model.load_state_dict(best_weights)
    model.eval()
    history["epochs_run"] = epoch
    history["stopped_early"] = epoch < epochs
    return model, history
