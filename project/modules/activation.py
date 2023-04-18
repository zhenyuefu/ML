import re
import numpy as np
from .module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._input_cache = x
        return np.tanh(x)

    def backward_update_gradient(self, input, delta):
        return delta

    def backward_delta(self, input, delta):
        return delta * (1 - np.tanh(input) ** 2)

    def update_parameters(self, gradient_step=1e-3):
        pass


class Sigmoid(Module):
    def forward(self, x):
        self._input_cache = x
        return 1 / (1 + np.exp(-x))

    def backward_update_gradient(self, input, delta):
        return delta

    def backward_delta(self, input, delta):
        yhat = self.forward(input)
        return delta * yhat * (1 - yhat)

    def update_parameters(self, gradient_step=1e-3):
        pass


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._input_cache = x
        return np.maximum(x, 0)

    def backward_update_gradient(self, input, delta):
        return delta

    def backward_delta(self, input, delta):
        return delta * (input > 0)

    def update_parameters(self, gradient_step=1e-3):
        pass


class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._input_cache = x
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)
        return softmax

    def backward_delta(self, input, delta):
        softmax = self.forward(input)
        return delta * (softmax * (1 - softmax))

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        return delta


class LogSoftMax(Module):
    def __init__(self):
        r"""
        LogSoftMax activation function
        log_softmax(x) = x - log(sum(exp(x)))
        """

        super().__init__()

    def forward(self, x):
        self._input_cache = x
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        log_softmax = x - np.log(np.sum(e_x, axis=1, keepdims=True))
        return log_softmax

    def backward_delta(self, input, delta):
        log_softmax = self.forward(input)
        softmax = np.exp(log_softmax)
        return delta - np.sum(delta * softmax, axis=1, keepdims=True) * softmax

    def backward_update_gradient(self, input, delta):
        return delta

    def update_parameters(self, gradient_step=1e-3):
        pass
