# Universal Approximation by Shallow Networks

This is the code we'll all use for our function experiments: affine,
exponential, polynomial, and sinusoidal. Pick your family, use the shared functions in `src/`, and keep your
experiments in your own notebook and branch.

## Getting started

You'll need Python 3.10+ and Git. Clone the repo from GitHub, then run these
commands one at a time in Terminal. Change the first path to wherever you
saved the project:

```bash
cd /Users/marina/projects/how-wide-is-enough
# Create and switch to our Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install what we need
python -m pip install -r requirements.txt
python -m ipykernel install --user --name universal-approximation-by-shallow-networks --display-name "Python (Universal Approximation by Shallow Networks)"
```

On Windows PowerShell, create the environment with `python -m venv .venv`
and activate it with `.venv\Scripts\Activate.ps1` instead.

## How to Run Your Experiment

Start in the project folder. Make a branch for your work, then open Jupyter.
Use your family name instead of `exponential`:

```bash
git switch -c experiment/exponential
cd notebooks
python -m notebook
```

Make a notebook in `notebooks/`, like `exponential.ipynb`. Pick
**Python (Universal Approximation by Shallow Networks)** as the kernel - the environment we just
created. **Shift+Enter** runs a cell; **Run All** runs everything in order.
Next time, just activate `.venv` and open Jupyter again.

Here's a starting point you can copy into your notebook. Change the function
and settings under `# Your settings`. Leave the folder check at the top;
it makes the shared imports work. For an example with shifts, have a look at
`shifted_exponential.ipynb`.

```python
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

if Path.cwd().name == "notebooks":
    os.chdir("..")

from src.functions import affine, exponential, polynomial, sinusoidal
from src.data import prepare_data, evaluation_grid
from src.experiments import run_experiments
from src.evaluate import predict, evaluate_model
from src.plots import plot_width_comparison, plot_test_predictions, plot_training_loss

# Your settings
function = exponential
domain = (-1.0, 1.0)
widths = [4, 16, 64]
learning_rates = [0.01, 0.05]
seeds = [21]

metrics_dir = Path("results/metrics") / function.__name__
figures_dir = Path("results/figures") / function.__name__
metrics_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Try each width and learning rate using training + validation data.
data = prepare_data(function, domain=domain)
x_train, y_train = data["train"]
x_val, y_val = data["val"]
runs, trained_runs = run_experiments(
    x_train, y_train, x_val, y_val, widths=widths,
    learning_rates=learning_rates, seeds=seeds,
    domain=domain, csv_path=metrics_dir / "tuning.csv",
)
summary = runs.groupby(["width", "learning_rate"], as_index=False)["validation_mse"].mean()
best = summary.sort_values(["validation_mse", "width", "learning_rate"]).iloc[0]

# Reuse the selected model from the sweep; no extra training needed.
# Use the first seed, fixed in advance, rather than picking the best seed.
key = (int(best["width"]), float(best["learning_rate"]), seeds[0])
model, history = trained_runs[key]
x_test, y_test = data["test"]
errors = evaluate_model(model, x_test, y_test)
print(errors)
pd.DataFrame([errors]).to_csv(metrics_dir / "test_metrics.csv", index=False)

# Save the plots.
fig = plot_width_comparison(runs)
fig.savefig(figures_dir / "width.png", dpi=150)
x_grid, y_grid = evaluation_grid(function, domain=domain)
fig = plot_test_predictions(x_grid, y_grid, x_test, predict(model, x_test))
fig.savefig(figures_dir / "predictions.png", dpi=150)
fig = plot_training_loss(history)
fig.savefig(figures_dir / "training.png", dpi=150)
plt.show()
```

Just a few things to remember:

- Start with `seeds = [21]` for quicker exploration. Later, try
  `seeds = [21, 22, 23]` to check whether the findings hold across different
  initial weights. The example selects settings by average validation MSE.
- Training uses the shared defaults, including early stopping. You can
  change `epochs` or `patience` later if needed.
- Put your choices and observations in your notebook. Reuse `src` for the
  shared code. If that code changes, **restart the kernel and Run All**.

Later, you can add `reduce_lr=True` to `run_experiments(...)` to reduce the
learning rate when validation MSE stalls. Leave it off while getting started.

## Sharing your results

Save your notebook, then open another terminal in the project folder:

```bash
git add notebooks/exponential.ipynb results/metrics/exponential/ results/figures/exponential/
git commit -m "Add exponential experiment"
git push -u origin experiment/exponential
```

Swap in your own notebook and family name. Then go to GitHub, click
**Compare & pull request**, choose `main` as the base, and write a few lines
about what you tried and found. Ask a teammate to review it before merging.

## Improving the shared code

Found a bug or want to tidy up something in `src/`? Make a separate branch
from the latest `main` first (save and commit any current work before switching):

```bash
git switch main
git pull
git switch -c fix/training
```

Use a name that describes your change, like `fix/training` or
`refactor/data`. Make your changes there, then commit and push:

```bash
git add src/
git commit -m "Fix training issue"
git push -u origin fix/training
```

Use your branch name in the push command. Open a pull request on GitHub
into `main`, briefly explain what changed, and ask teammates to review it.
