"""Plots for test performance and model behavior across hidden-layer widths."""

import matplotlib.pyplot as plt
import numpy as np

from .model import predict


def _choose_widths(available_widths, widths_to_plot):
    available = set(available_widths)
    if widths_to_plot is None:
        selected = tuple(sorted(available))
    else:
        selected = tuple(widths_to_plot)
    if not selected:
        raise ValueError("at least one width must be available to plot")
    if any(not isinstance(width, (int, np.integer)) or width < 1 for width in selected):
        raise ValueError("widths_to_plot must contain positive integers")
    if len(set(selected)) != len(selected):
        raise ValueError("widths_to_plot must not contain duplicates")
    missing = set(selected) - available
    if missing:
        raise ValueError(f"no fitted results are available for widths: {sorted(missing)}")
    return tuple(sorted(selected))


def _test_arrays(x_test, y_test):
    x = np.asarray(x_test, dtype=float).ravel()
    y = np.asarray(y_test, dtype=float).ravel()
    if x.size == 0 or x.shape != y.shape:
        raise ValueError("x_test and y_test must have matching, nonempty shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("test values must be finite")
    return x, y, np.argsort(x)


def plot_test_error_by_width(results, widths_to_plot=None, log_scale=False,
                             title="Test maximum error by hidden-layer width"):
    """Plot test maximum absolute error against width for selected networks.

    results is the DataFrame created in the experiment notebook. log_scale
    applies to both axes. Use test errors for reporting, not model selection.
    """
    required = {"model", "width", "test_max_error"}
    if not required.issubset(results.columns):
        raise ValueError("results must contain model, width, and test_max_error")
    network_rows = results[results["model"] == "sigmoid network"]
    if network_rows.empty:
        raise ValueError("results contains no sigmoid-network rows")

    widths = tuple(int(width) for width in network_rows["width"])
    errors = np.asarray(network_rows["test_max_error"], dtype=float)
    if len(set(widths)) != len(widths):
        raise ValueError("results must contain one sigmoid-network row per width")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("test errors must be finite and nonnegative")
    selected = _choose_widths(widths, widths_to_plot)
    error_by_width = dict(zip(widths, errors))
    plotted_errors = [error_by_width[width] for width in selected]
    if log_scale and any(error <= 0 for error in plotted_errors):
        raise ValueError("log scale requires positive test errors")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(selected, plotted_errors, marker="o", linewidth=1.5)
    ax.set_xticks(selected, labels=[str(width) for width in selected])
    ax.set(title=title, xlabel="Hidden neurons (N)",
           ylabel="Test maximum absolute error")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    return fig


def plot_predictions_vs_actual(fitted_models, x_test, y_test, widths_to_plot=None,
                               title="Predictions and actual test data"):
    """Overlay selected model predictions and actual test values against x."""
    x, actual, order = _test_arrays(x_test, y_test)
    widths = _choose_widths(fitted_models, widths_to_plot)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.scatter(x[order], actual[order], color="black", s=10, alpha=0.45,
               label="Actual test data")
    for idx, width in enumerate(widths):
        prediction = predict(fitted_models[width]["parameters"], x)
        if not np.isfinite(prediction).all():
            raise ValueError(f"predictions for width {width} must be finite")
        ax.plot(x[order], prediction[order], color=f"C{idx}", linewidth=1.5,
                label=f"Width {width}")

    ax.set(title=title, xlabel="x", ylabel="Function value")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_signed_residuals(fitted_models, x_test, y_test, widths_to_plot=None,
                          title="Signed residuals: prediction - actual"):
    """Plot selected models' prediction-minus-actual residuals against x.

    The legend reports each width's mean squared residual over the test data.
    """
    x, actual, order = _test_arrays(x_test, y_test)
    widths = _choose_widths(fitted_models, widths_to_plot)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    for idx, width in enumerate(widths):
        prediction = predict(fitted_models[width]["parameters"], x)
        if not np.isfinite(prediction).all():
            raise ValueError(f"predictions for width {width} must be finite")
        residual = prediction - actual
        mse = np.mean(residual ** 2)
        ax.plot(x[order], residual[order], color=f"C{idx}", linewidth=1.5,
                label=f"Width {width} (MSE={mse:.2e})")

    ax.set(title=title, xlabel="x", ylabel="Residual")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    return fig


def plot_validation_convergence(fitted_models, widths_to_plot=None,
                                title="Validation error over BFGS iterations"):
    """Plot validation maximum error per BFGS iteration for selected widths."""
    widths = _choose_widths(fitted_models, widths_to_plot)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for idx, width in enumerate(widths):
        history = np.asarray(
            fitted_models[width]["validation_max_error_history"], dtype=float
        )
        if history.size == 0 or not np.isfinite(history).all():
            raise ValueError(f"validation history for width {width} must be nonempty and finite")
        ax.plot(np.arange(history.size), history, color=f"C{idx}",
                linewidth=1.5, label=f"Width {width}")

    ax.set(title=title, xlabel="BFGS iteration",
           ylabel="Validation maximum absolute error")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig
