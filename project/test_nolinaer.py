from matplotlib import pyplot as plt
import numpy as np

from modules import Linear, TanH, Sigmoid, ReLU, MSELoss, CrossEntropyLoss
from utils.mltools import (
    gen_arti,
    plot_frontiere_3d,
    plot_data_3d,
    plot_data,
    plot_frontiere,
)


# Définir les paramètres d'entrée et de sortie
input_size = 2
hidden_size = 10
output_size = 1

# Définir les données d'entraînement
X, Y = gen_arti(data_type=1)

# Initialiser les modules
# w1 = np.ones((input_size, hidden_size))
linear1 = Linear(input_size, hidden_size)
tanh = TanH()
# w2 = np.asarray([1.0, -2.0]).reshape(hidden_size, output_size)
linear2 = Linear(hidden_size, output_size)
sigmoid = Sigmoid()
celoss = CrossEntropyLoss()

# Boucle d'entraînement
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    l1 = linear1.forward(X)
    tan = tanh.forward(l1)
    l2 = linear2.forward(tan)
    yhat = sigmoid.forward(l2)

    # Calculer la perte
    loss = celoss.forward(Y, yhat)
    if epoch % 1000 == 0:
        print("Epoch %d: Loss = %f" % (epoch, loss.mean()))

    # Backward pass
    delta = celoss.backward(Y, yhat)
    delta = sigmoid.backward_delta(l2, delta)
    linear2.backward_update_gradient(tan, delta)
    delta = linear2.backward_delta(tan, delta)
    delta = tanh.backward_delta(l1, delta)
    linear1.backward_update_gradient(X, delta)

    # Mettre à jour les paramètres
    linear1.update_parameters()
    linear2.update_parameters()

    # Remettre à zéro les gradients
    linear1.zero_grad()
    linear2.zero_grad()

# Afficher les résultats
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_frontiere_3d(
    ax,
    X,
    lambda x: sigmoid.forward(linear2.forward(tanh.forward(linear1.forward(x)))),
    step=100,
)

plot_data_3d(ax, X, Y)

plt.show()

plot_frontiere(
    X,
    lambda x: sigmoid.forward(linear2.forward(tanh.forward(linear1.forward(x)))),
    step=100,
)

plot_data(X, Y)
plt.show()
