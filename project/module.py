import numpy as np
import torch.nn


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def forward(self, data):
        raise NotImplementedError

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        for param, grad in zip(self._parameters, self._gradient):
            param -= gradient_step * grad

    def zero_grad(self):
        self._gradient = None

    def parameters(self):
        return self._parameters


class Linear(Module):
    def __init__(self, input_size, output_size, w=None, b=None, bias=True):
        super(Linear, self).__init__()
        if w is None:
            self.w = np.random.randn(input_size, output_size)
        else:
            self.w = w
        if bias:
            if b is None:
                self.b = np.ones(output_size)
            else:
                self.b = b
        else:
            self.b = np.zeros(output_size)
        self._parameters = [self.w, self.b]

    def forward(self, x):
        yhat = x @ self.w + self.b
        return yhat

    def backward_update_gradient(self, input, delta):
        dw = input.T @ delta
        db = delta.sum(axis=0)
        self._gradient = [dw, db]

    def backward_delta(self, input, delta):
        dinput = delta @ self.w.T
        return dinput


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
