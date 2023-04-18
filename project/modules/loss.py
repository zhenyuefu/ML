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
        return 2 * (yhat - y) / y.shape[0]


class CrossEntropyLoss(Loss):
    def forward(self, y, yhat):
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        loss = -np.sum(y * (yhat - log_sum_exp), axis=1)
        return np.mean(loss)

    def backward(self, y, yhat):
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        softmax = np.exp(yhat - log_sum_exp)
        return -(y - softmax) / y.shape[0]
