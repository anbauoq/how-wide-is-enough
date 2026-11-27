"""Target families with shifts: f(x - horizontal_shift) + vertical_shift.

Positive horizontal shifts move the graph right; positive vertical shifts
move it up. Defaults leave each original function unchanged. All targets
accept NumPy arrays and preserve their shape.
"""

import numpy as np

def affine(x, horizontal_shift=0.0, vertical_shift=0.0):
    """Return 2(x - horizontal_shift) + 1 + vertical_shift, with x's shape."""
    return 2 * (x - horizontal_shift) + 1 + vertical_shift


def exponential(x, horizontal_shift=0.0, vertical_shift=0.0):
    """Return exp(x - horizontal_shift) + vertical_shift, with x's shape."""
    return np.exp(x - horizontal_shift) + vertical_shift


def polynomial(x, horizontal_shift=0.0, vertical_shift=0.0):
    """Return (x - horizontal_shift)^3 - (x - horizontal_shift) + vertical_shift.

    The result has x's shape.
    """
    shifted_x = x - horizontal_shift
    return shifted_x**3 - shifted_x + vertical_shift


def sinusoidal(x, horizontal_shift=0.0, vertical_shift=0.0):
    """Return sin(2 pi (x - horizontal_shift)) + vertical_shift, with x's shape."""
    return np.sin(2 * np.pi * (x - horizontal_shift)) + vertical_shift
