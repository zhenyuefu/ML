import numpy as np
import pickle as pkl
from modules.container import Sequential
from modules.conv import Conv2d
from modules.flatten import Flatten
from modules.pooling import MaxPool2d
from modules.linear import Linear
from modules.activation import ReLU
from modules.loss import CrossEntropyLoss
from optim.optimizer import Optimizer
from optim.sgd import SGD

# Load data
data = pkl.load(open("data/usps.pkl", "rb"))

X_train = np.array(data["X_train"], dtype=float).reshape(-1, 1, 16, 16)
X_test = np.array(data["X_test"], dtype=float).reshape(-1, 1, 16, 16)
Y_train = data["Y_train"]
Y_test = data["Y_test"]


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

# Create LeNet model
net = Sequential(
    Conv2d(1, 6, kernel_size=2),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(6, 16, kernel_size=2),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(144, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, 10),
)

loss = CrossEntropyLoss()
optimizer = Optimizer(net, loss, learning_rate=1e-2)

# Train the model
SGD(X_train, Y_train, batch_size=64, num_iterations=100, optimizer=optimizer)

# Compute network output on test set
output = net.forward(X_test)

# Compute accuracy
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
print(f"Accuracy: {accuracy}")
