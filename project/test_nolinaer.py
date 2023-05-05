from matplotlib import pyplot as plt

from modules import Linear, TanH, Sigmoid, CrossEntropyLoss
from modules.loss import BCELoss
from utils.mltools import (
    gen_arti,
    plot_frontiere_3d,
    plot_data_3d,
    plot_data,
    plot_frontiere,
)


# Définir les paramètres d'entrée et de sortie
input_size = 2
hidden_size = 13
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
celoss = BCELoss()
losses = []

# Boucle d'entraînement
num_epochs = 100000
for epoch in range(num_epochs):
    # Forward pass
    l1 = linear1.forward(X)
    tan = tanh.forward(l1)
    l2 = linear2.forward(tan)
    yhat = sigmoid.forward(l2)

    # Calculer la perte
    loss = celoss.forward(Y, yhat)
    if epoch % 2000 == 0:
        print("Epoch %d: Loss = %f" % (epoch, loss.mean()))
    losses.append((epoch, loss.mean()))

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
fig = plt.figure(figsize=(15, 5))
fig.suptitle("Non-linear classification")

ax = fig.add_subplot(131, projection="3d")
fig.subplots_adjust(
    left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2
)
plot_frontiere_3d(
    ax,
    X,
    lambda x: sigmoid.forward(linear2.forward(tanh.forward(linear1.forward(x)))),
    step=100,
)

plot_data_3d(ax, X, Y)

fig.add_subplot(132, aspect="equal")

plot_frontiere(
    X,
    lambda x: sigmoid.forward(linear2.forward(tanh.forward(linear1.forward(x)))),
    step=100,
)

plot_data(X, Y)

fig.add_subplot(133)

plt.plot([x[0] for x in losses], [x[1] for x in losses], "-")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("loss")
plt.savefig("test_nonlinear.png")
