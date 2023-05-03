import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle as pkl
import torch.nn as nn


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

# Create LeNet model
net = nn.Sequential(
    nn.Conv1d(1, 32, kernel_size=2),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(4064, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)

X = torch.rand(size=(1, 1, 16 * 16), dtype=torch.float32)
for layer in net:
    X = layer(X)
    params = list(layer.parameters())
    for p in params:
        print(p.shape)
    print(layer.__class__.__name__, f"output shape: {X.shape}")

# Convert data to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_t, Y_test_t)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the loss function and the optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training parameters
epochs = 40

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Test the model
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == torch.max(target, 1)[1]).sum().item()

print(f"Accuracy of the model on the test set: {100 * correct / total}%")
