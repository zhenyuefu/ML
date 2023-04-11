import numpy as np
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, w=None, b=None, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = 2 * np.random.rand(in_features, out_features) - 1
        if bias:
            self.bias = 2 * np.random.rand(out_features) - 1
            self._parameters = [self.weight, self.bias]
        else:
            self.bias = None
            self._parameters = [self.weight]

    def forward(self, x):
        if self.bias is None:
            yhat = x @ self.weight
        else:
            yhat = x @ self.weight + self.bias
        return yhat

    def backward_update_gradient(self, input, delta):
        dw = np.matmul(input.T, delta)
        if self.bias is None:
            db = None
            self._gradient = [dw]
        else:
            db = np.matmul(delta.T, np.ones(input.shape[0]))
            self._gradient = [dw, db]

    def backward_delta(self, input, delta):
        return np.matmul(delta, self.weight.T)
