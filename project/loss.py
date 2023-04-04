import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.mean((yhat - y) ** 2, axis=1)

    def backward(self, y, yhat):
        return 2 * (yhat - y) / y.shape[1]


class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat), axis=1)
        return loss

    def backward(self, y, yhat):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)
        return (yhat - y) / (yhat * (1 - yhat))
