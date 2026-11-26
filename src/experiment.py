"""Data generation, test evaluation, and regression baselines."""

import numpy as np

from .model import predict


def generate_data(target, n_train=1000, n_validation=1000, n_test=10000,
                  domain=(-1.0, 1.0), seed=21):
    """Return separate ``(x, y)`` pairs sampled uniformly over ``domain``."""
    # Use one seeded generator so everyone can recreate the same three splits.
    rng = np.random.default_rng(seed)
    low, high = domain
    data = {}

    for name, count in (
        ("train", n_train),
        ("validation", n_validation),
        ("test", n_test),
    ):
        x = np.sort(rng.uniform(low, high, count))
        data[name] = (x, np.asarray(target(x), dtype=float))

    return data


def evaluate_model(parameters, x_test, y_test):
    """Return the maximum absolute error on the supplied test points."""
    # Test data is only for the final score, never for training or selection.
    predictions = predict(parameters, x_test)
    return float(np.max(np.abs(predictions - y_test)))


def regression_baselines(x_train, y_train, x_test, y_test):
    """Fit degree-1 and degree-3 polynomials; return test maximum errors."""
    results = []
    # Fit both baselines on the same training points as the neural network.
    for name, degree in (("linear", 1), ("cubic polynomial", 3)):
        coefficients = np.polynomial.polynomial.polyfit(x_train, y_train, degree)
        prediction = np.polynomial.polynomial.polyval(x_test, coefficients)
        results.append({
            "model": name,
            "test_max_error": float(np.max(np.abs(prediction - y_test))),
        })
    return results
