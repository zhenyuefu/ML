import numpy as np
from .module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)

    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        yhat = self.forward(input)
        return delta * yhat * (1 - yhat)

    def update_parameters(self, gradient_step=1e-3):
        pass


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(x, 0)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (input > 0)

    def update_parameters(self, gradient_step=1e-3):
        pass
