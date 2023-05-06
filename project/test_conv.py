import numpy as np
import pickle as pkl
from modules.container import Sequential
from modules.conv import Conv1d
from modules.flatten import Flatten
from modules.pooling import MaxPool1d
from modules.linear import Linear
from modules.activation import ReLU
from modules.loss import CrossEntropyLoss
from optim.optimizer import Optimizer
from optim.sgd import SGD
from utils.plot import plot_loss

# Parameters
learning_rate = 1e-2
num_iterations = 150
batch_size = 64

# Load data
data = pkl.load(open("data/usps.pkl", "rb"))

X_train = np.array(data["X_train"], dtype=float).reshape(-1, 1, 16 * 16)
X_test = np.array(data["X_test"], dtype=float).reshape(-1, 1, 16 * 16)
Y_train = data["Y_train"]
Y_test = data["Y_test"]


def one_hot(y):
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


Y_train = one_hot(Y_train)
Y_test = one_hot(Y_test)

net = Sequential(
    Conv1d(1, 32, kernel_size=3),
    MaxPool1d(2, 2),
    Flatten(),
    Linear(4064, 100),
    ReLU(),
    Linear(100, 10),
)

loss = CrossEntropyLoss()

optimizer = Optimizer(net, loss, lr=learning_rate)

loss_list = SGD(
    X_train,
    Y_train,
    batch_size=batch_size,
    num_iterations=num_iterations,
    optimizer=optimizer,
)

# Compute network output on test set
net.eval()
output = net.forward(X_test)

# Compute accuracy
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
print(f"Accuracy: {accuracy}")
# 150 iter loss:0.0672 acc:0.9322

# Plot loss
plot_loss(
    loss_list,
    accuracy,
    batch_size,
    num_iterations,
    learning_rate,
    "Conv1d USPS",
)
