# Universal Approximation by Shallow Networks

The shared code is in `src/`: `functions.py` defines the target families,
`model.py` handles network prediction and training, `experiment.py` handles
data generation, test evaluation, and baselines, and `plots.py` provides
visualizations. Each team member can copy `notebooks/template.ipynb`, select
their target and parameters, and save metrics and figures in `results/`.

## Setup

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

## Shared protocol

- Sample 1,000 training points, 1,000 validation points, and 10,000 test
  points independently and uniformly from `[-1, 1]`.
- Train one-hidden-layer logistic-sigmoid networks with BFGS to minimize
  training MSE. Use widths 1, 2, 4, 8, 16, 32, 64, 128, and 256.
- Use 10 random starts per width. Early-stop after 100 iterations without a
  validation maximum-error improvement of `1e-8`, up to 1,500 iterations.
  Keep each start's best-validation checkpoint and select the start with the
  smallest training-loss gradient norm.
- Report maximum absolute test error and compare with linear and cubic
  polynomial regression fitted to the same training points.

`src/functions.py` defines parameterized target families. Its defaults are
affine `2x + 1`, cubic `x³ - x`, sinusoidal `sin(2πx)`, and exponential
`exp(x)`.
