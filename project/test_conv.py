import numpy as np
import pickle as pkl
from modules.container import Sequential
from modules.conv import Conv1D
from modules.pooling import MaxPool1D
from modules.conv import Flatten
from modules.linear import Linear
from modules.activation import ReLU
from modules.loss import CrossEntropyLoss
from optim.optimizer import Optimizer
from optim.sgd import SGD

# Load data
data = pkl.load(open("data/usps.pkl", "rb"))

X_train = np.array(data["X_train"], dtype=float).reshape(-1, 16 * 16, 1)
X_test = np.array(data["X_test"], dtype=float).reshape(-1, 16 * 16, 1)
Y_train = data["Y_train"]
Y_test = data["Y_test"]


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

net = Sequential(
    Conv1D(3, 1, 32),
    MaxPool1D(2, 2),
    Flatten(),
    Linear(4064, 100),
    ReLU(),
    Linear(100, 10),
)

loss = CrossEntropyLoss()

optimizer = Optimizer(net, loss, learning_rate=2e-2)

SGD(X_train, Y_train, batch_size=32, num_iterations=150, optimizer=optimizer)

# Compute network output on test set
output = net.forward(X_test)

# Compute accuracy
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
print(f"Accuracy: {accuracy}")
# Loss = 0.0921 Accuracy = 0.926
