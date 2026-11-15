"""A single-hidden-layer neural network for scalar functions."""

from torch import nn


class ShallowNetwork(nn.Module):
    """Map one scalar input to one output through width hidden neurons.

    activation is "sigmoid" or "tanh". Input/output tensors have shape
    (number_of_points, 1). train_model creates this network for you.
    """

    def __init__(self, width=32, activation="sigmoid"):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in activations:
            raise ValueError("activation must be 'sigmoid' or 'tanh'")
        self.network = nn.Sequential(
            nn.Linear(1, width), activations[activation](), nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.network(x)
