import numpy as np
import pickle as pkl
from modules.container import Sequential
from modules.linear import Linear
from modules.activation import ReLU, TanH
from modules.loss import CrossEntropyLoss
from optim.optimizer import Optimizer
from optim.sgd import SGD
from utils.plot import plot_loss


data = pkl.load(open("data/usps.pkl", "rb"))

X_train = np.array(data["X_train"], dtype=float)
X_test = np.array(data["X_test"], dtype=float)
Y_train = data["Y_train"]
Y_test = data["Y_test"]


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)


net = Sequential(
    Linear(256, 100),
    ReLU(),
    Linear(100, 10),
)

loss = CrossEntropyLoss()

learning_rate = 0.02
batch_size = 32
num_iterations = 150

optimizer = Optimizer(net, loss, lr=learning_rate)


loss_list = SGD(
    X_train,
    Y_train,
    batch_size=batch_size,
    num_iterations=num_iterations,
    optimizer=optimizer,
)


# Compute network output on test set
output = net.forward(X_test)

# Compute accuracy
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
print(f"Accuracy: {accuracy}")
# Accuracy: 0.9094 (with 500 iterations) Loss = 0.00451

# Plot loss
plot_loss(
    loss_list, accuracy, batch_size, num_iterations, learning_rate, "usps_loss.png"
)
