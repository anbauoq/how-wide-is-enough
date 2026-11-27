"""Reproducible training, validation, and test data."""

import numpy as np


def _check_domain(domain):
    low, high = domain
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("domain must contain two finite bounds with low < high")
    return low, high


def generate_data(target, n_samples=200, domain=(-1.0, 1.0), seed=21):
    """Sample x uniformly in domain and calculate y = target(x).

    target must return one value per input. Returns (x, y), two NumPy
    arrays of shape (n_samples, 1). The same seed gives the same samples.
    """
    if not callable(target):
        raise TypeError("target must be a function, for example exponential")

    low, high = _check_domain(domain)
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    x = rng.uniform(low, high, size=(n_samples, 1))
    return x, np.asarray(target(x), dtype=float).reshape(x.shape)

generate_data(affine)


def prepare_data(target, domain=(-1.0, 1.0), n_train=1000, n_val=200,
                 n_test=500, seed=21):
    """Return separate train, val, and test datasets in a dictionary.

    Each value is an (x, y) pair of NumPy column arrays. Counts are independent.
    Reuse train/val for all candidates; evaluate test only after selection.
    The same seed and settings reproduce the same data.
    """
    if any(not isinstance(size, int) or size < 1 for size in (n_train, n_val, n_test)):
        raise ValueError("all sample counts must be positive integers")
    x, y = generate_data(target, n_train + n_val + n_test, domain, seed)
    end_val = n_train + n_val
    return {
        "train": (x[:n_train], y[:n_train]),
        "val": (x[n_train:end_val], y[n_train:end_val]),
        "test": (x[end_val:], y[end_val:]),
    }


def evaluation_grid(target, n_points=1000, domain=(-1.0, 1.0)):
    """Return (x, y) on an evenly spaced grid, including both endpoints.

    Both NumPy arrays have shape (n_points, 1). Use this grid to draw the
    true function, separately from random training/validation/test data.
    """
    if not callable(target):
        raise TypeError("target must be a function, for example exponential")
    low, high = _check_domain(domain)
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")
    x = np.linspace(low, high, n_points).reshape(-1, 1)
    return x, np.asarray(target(x), dtype=float).reshape(x.shape)
