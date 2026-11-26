"""Parameterized target families used in the approximation experiments."""

import numpy as np


# Tweak these parameters to make a different member of the same function family.
def affine(x, slope=2.0, intercept=1.0):
    """Return slope * x + intercept."""
    return slope * x + intercept


def polynomial(x, cubic=1.0, quadratic=0.0, linear=-1.0, constant=0.0):
    """Return cubic*x**3 + quadratic*x**2 + linear*x + constant."""
    return cubic * x**3 + quadratic * x**2 + linear * x + constant


def sinusoidal(x, amplitude=1.0, frequency=2 * np.pi, phase=0.0, offset=0.0):
    """Return amplitude * sin(frequency*x + phase) + offset."""
    return amplitude * np.sin(frequency * x + phase) + offset


def exponential(x, amplitude=1.0, rate=1.0, offset=0.0):
    """Return amplitude * exp(rate*x) + offset."""
    return amplitude * np.exp(rate * x) + offset
