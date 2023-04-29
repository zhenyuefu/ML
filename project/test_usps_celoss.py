import numpy as np
import pickle as pkl
from modules.container import Sequential
from modules.linear import Linear
from modules.activation import Sigmoid, TanH
from modules.loss import CrossEntropyLoss
from optim.sgd import SGD


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
    TanH(),
    Linear(100, 10),
)

loss = CrossEntropyLoss()

optimizer = SGD(net, loss, learning_rate=0.2)


for epoch in range(15000):
    # sample random minibatch
    idx = np.random.choice(X_train.shape[0], 150, replace=False)
    X_batch = X_train[idx]
    Y_batch = Y_train[idx]
    loss_value = optimizer.step(X_batch, Y_batch)
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, Loss: {loss_value}")


# Compute network output on test set
output = net.forward(X_test)

# Compute accuracy
accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(Y_test, axis=1))
print(f"Accuracy: {accuracy}")
