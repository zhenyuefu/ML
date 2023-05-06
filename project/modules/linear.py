import numpy as np
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._parameters["weight"] = 2 * np.random.rand(in_features, out_features) - 1
        if bias:
            self._parameters["bias"] = np.zeros(out_features, dtype=float)
        else:
            self._parameters["bias"] = None

    def forward(self, x):
        if self.is_training:
            self._input_cache = x
        yhat = np.matmul(x, self._parameters["weight"])
        if self._parameters["bias"] is not None:
            yhat = yhat + self._parameters["bias"]
        return yhat

    def backward_update_gradient(self, input, delta):
        dw = np.matmul(input.T, delta)
        self._gradient["weight"] = dw
        if self._parameters["bias"] is not None:
            db = np.matmul(delta.T, np.ones(input.shape[0], dtype=float))
            self._gradient["bias"] = db

    def backward_delta(self, input, delta):
        return np.matmul(delta, self._parameters["weight"].T)

    def set_weight(self, weight):
        self._parameters["weight"] = weight
