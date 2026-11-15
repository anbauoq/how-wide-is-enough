"""Simple plots for parameter selection, training, and test predictions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_width_comparison(results):
    """Plot mean validation MSE against width, one curve per learning rate.

    Accepts the DataFrame from run_experiments and returns a Matplotlib
    Figure. Required columns: domain, width, learning_rate, validation_mse.
    Positive MSE uses a log scale; zero MSE uses a linear scale instead.
    """
    if results.empty:
        raise ValueError("results is empty; run an experiment before plotting")
    required = {"domain", "width", "learning_rate", "validation_mse"}
    if not required.issubset(results.columns):
        raise ValueError("results must contain domain, width, learning_rate, and validation_mse")
    values = results["validation_mse"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("MSE values must be finite and nonnegative")

    domains = results["domain"].unique()
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 4.5),
                             squeeze=False, layout="constrained")
    for ax, domain in zip(axes.ravel(), domains):
        rows = results[results.domain == domain]
        for rate, group in rows.groupby("learning_rate"):
            means = group.groupby("width")["validation_mse"].mean()
            ax.plot(means.index, means, marker="o", label=f"{rate:g}")
        ax.set_xscale("log", base=2)
        widths = sorted(rows.width.unique())
        ax.set_xticks(widths, labels=[str(width) for width in widths])
        if (rows.validation_mse > 0).all():
            ax.set_yscale("log")
            ylabel = "Validation MSE (log scale)"
        else:
            ax.set_ylim(bottom=0)
            ylabel = "Validation MSE"
        ax.minorticks_off()
        title = "Experiment" if domain == "Experiment" else f"Interval {domain}"
        ax.set(title=title, xlabel="Hidden neurons", ylabel=ylabel)
        ax.legend(title="Learning rate")
        ax.grid(alpha=0.3)
    return fig


def plot_test_predictions(x_grid, y_grid, x_test, y_pred, title="Final test predictions",
                          x_limits=None):
    """Plot only the true function and predictions at held-out test inputs.

    x_grid/y_grid and x_test/y_pred are matching pairs of arrays, shape
    (n, 1) or (n,). x_limits optionally selects a close-up with its own
    y scale. Returns a Matplotlib Figure.
    """
    x = np.asarray(x_grid).ravel()
    truth = np.asarray(y_grid).ravel()
    test_x = np.asarray(x_test).ravel()
    prediction = np.asarray(y_pred).ravel()
    if x_limits is not None:
        low, high = x_limits
        grid_mask = (x >= low) & (x <= high)
        test_mask = (test_x >= low) & (test_x <= high)
        x, truth = x[grid_mask], truth[grid_mask]
        test_x, prediction = test_x[test_mask], prediction[test_mask]
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.axhline(0, color="lightgray")
    ax.axvline(0, color="lightgray")
    ax.plot(x[order], truth[order], color="black", label="True function")
    ax.scatter(test_x, prediction, color="orange", s=15, alpha=0.7,
               label="Predicted test points")
    ax.set(title=title, xlabel="x", ylabel="Function value")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_training_loss(history, title="Training and validation MSE"):
    """Plot the history from train_model; return a Matplotlib Figure.

    Shows training_mse, validation_mse, and the best_epoch checkpoint,
    with the full run and a closer view of the final 20% of updates.
    """
    training = history["training_mse"]
    validation = history["validation_mse"]
    epochs = np.arange(1, len(training) + 1)
    start = int(len(training) * 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.suptitle(title)
    for ax, first, subtitle in zip(axes, (0, start), ("Full training", "Final 20%")):
        ax.plot(epochs[first:], training[first:], label="Training")
        ax.plot(epochs[first:], validation[first:], linestyle="--", label="Validation")
        if history["best_epoch"] >= epochs[first]:
            ax.axvline(history["best_epoch"], color="gray", linestyle=":", label="Restored epoch")
        ax.set(title=subtitle, xlabel="Epoch", ylabel="MSE")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend()
    return fig


def plot_metrics(metrics, title="Final training and test errors"):
    """Compare a metrics DataFrame in separate panels for different units.

    Required columns: split (such as Training or Test), mse, rmse,
    max_absolute_error. Returns a Matplotlib Figure.
    """
    columns = ("mse", "rmse", "max_absolute_error")
    titles = ("MSE (squared units)", "RMSE (function units)", "Maximum absolute error")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    fig.suptitle(title)
    for ax, column, subtitle in zip(axes, columns, titles):
        values = metrics[column].to_numpy()
        bars = ax.bar(metrics["split"], values)
        ax.set_title(subtitle)
        ax.set_ylim(0, max(float(values.max()) * 1.25, 1e-12))
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center")
    return fig
