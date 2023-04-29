import numpy as np
from numpy import ndarray


class Loss(object):
    def forward(self, y, yhat):
        raise NotImplementedError

    def backward(self, y, yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y: ndarray, yhat: ndarray):
        return np.mean((yhat - y) ** 2, axis=1)

    def backward(self, y: ndarray, yhat: ndarray):
        return 2 * (yhat - y) / y.shape[0]


class CrossEntropyLoss(Loss):
    def forward(self, y: ndarray, yhat: ndarray):
        r"""
        Computes the cross-entropy loss between y and yhat.
        Cross-entropy loss is defined as:
        :math:`L = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^M y_{ij} \log(softmax(\hat{y}_{ij}))`

        where :math:`N` is the batch size, :math:`M` is the number of classes,

        Args:
            y (array like): one-hot encoded labels of shape (batch_size, num_classes)
            yhat (array like): predicted probabilities of shape (batch_size, num_classes)
        """
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        loss = -np.sum(y * (yhat - log_sum_exp), axis=1)
        return np.mean(loss)

    def backward(self, y: ndarray, yhat: ndarray):
        r"""
        Computes the gradient of the cross-entropy loss with respect to yhat.
        The gradient is defined as:
        :math:`\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{e^{\hat{y}}}{\sum_{j=1}^M e^{\hat{y}_j}}`

        Args:
            y (array like): one-hot encoded labels of shape (batch_size, num_classes)
            yhat (array like): predicted probabilities of shape (batch_size, num_classes)
        """
        yhat_max = np.max(yhat, axis=1, keepdims=True)
        log_sum_exp = yhat_max + np.log(
            np.sum(np.exp(yhat - yhat_max), axis=1, keepdims=True)
        )
        softmax = np.exp(yhat - log_sum_exp)
        return -(y - softmax) / y.shape[0]
