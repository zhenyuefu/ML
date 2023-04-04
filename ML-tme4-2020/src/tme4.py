import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# def hor_to_lin(x):
#     return np.array([[xi] for xi in x])


# Perceptron loss function
def perceptron_loss(w, x, y):
    return np.maximum(0, -y * np.dot(x, w))


# Perceptron gradient function
def perceptron_grad(w, x, y):
    y_ = np.where(y <= 0, -1, 1)
    scores = np.dot(x, w)
    mistakes = np.where(y_ * scores <= 0, 1, 0)
    grad_weights = -np.dot(x.T, y_ * mistakes) / x.shape[0]
    return grad_weights


class Lineaire(object):
    def __init__(
        self, loss=perceptron_loss, loss_g=perceptron_grad, max_iter=100, eps=0.01
    ):
        self.max_iter, self.eps = max_iter, eps
        self.w = None
        self.loss, self.loss_g = loss, loss_g

    def fit(self, datax, datay, batch_size=None):
        n_samples, n_features = datax.shape
        # Initialize the weights
        self.w = np.zeros(n_features + 1)
        datax = np.concatenate((datax, np.ones((n_samples, 1))), axis=1)

        # Initialize the cost history list
        self.cost_history = []

        # Perform gradient descent for max_iter iterations
        for epoch in range(self.max_iter):
            if batch_size is None:
                # Batch gradient descent
                batch_x, batch_y = datax, datay
            elif batch_size == 1:
                # Stochastic gradient descent
                indices = np.random.permutation(n_samples)
                batch_x, batch_y = datax[indices], datay[indices]
            else:
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)[:batch_size]
                batch_x, batch_y = datax[indices], datay[indices]

            # Calculate the gradient
            gradient = self.loss_g(self.w, batch_x, batch_y)

            # Update the weights
            self.w -= self.eps * gradient

            # Calculate the cost
            cost = np.mean(self.loss(self.w, datax, datay))
            print(f"epoch: {epoch}, cost: {cost}")

            # Append the cost to the cost history list
            self.cost_history.append(cost)

    def predict(self, datax):
        if self.w is None:
            raise Exception("Model not trained yet")
        n_samples = datax.shape[0]
        datax = np.concatenate((datax, np.ones((n_samples, 1))), axis=1)
        return np.sign(np.dot(datax, self.w))

    def score(self, datax, datay):
        # Calculate the predictions for the given datax
        predictions = self.predict(datax)
        # Calculate the percentage of correct classifications
        correct_classifications = np.mean(predictions == datay)

        return correct_classifications


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def get_usps(l, datax, datay):
    if type(l) != list:
        resx = datax[datay == l, :]
        resy = datay[datay == l]
        return resx, resy
    tmp = list(zip(*[get_usps(i, datax, datay) for i in l]))
    tmpx, tmpy = np.vstack(tmp[0]), np.hstack(tmp[1])
    return tmpx, tmpy


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")


if __name__ == "__main__":
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax, datay = get_usps([neg, pos], alltrainx, alltrainy)
    testx, testy = get_usps([neg, pos], alltestx, alltesty)
    datay = np.where(datay == neg, -1, 1)  # 5 -> -1, 6 -> 1
    testy = np.where(testy == neg, -1, 1)  # 5 -> -1, 6 -> 1
    model = Lineaire()
    model.fit(datax, datay, batch_size=10)
    print(model.score(testx, testy))
    if model.w is not None:
        show_usps(model.w[:-1])
        plt.show()
