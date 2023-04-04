import numpy as np
from loss import MSELoss

from module import Linear

# use sklearn to generate data
from sklearn.datasets import make_regression

# Création des données
X, y, coef = make_regression(
    n_samples=100, n_features=5, noise=10, random_state=42, coef=True
)

y = y.reshape(-1, 1)

# Instanciation du module Linear et de la fonction de coût MSELoss
linear = Linear(5, 1)
loss_fn = MSELoss()

# Boucle d'apprentissage
for i in range(1000):
    # Forward pass
    y_pred = linear.forward(X)
    loss = loss_fn.forward(y, y_pred)

    # Backward pass
    delta = loss_fn.backward(y, y_pred)
    linear.backward_update_gradient(X, delta)

    # Mise à jour des paramètres
    linear.update_parameters()
    linear.zero_grad()

    # Affichage de la perte
    if i % 10 == 0:
        print(f"Étape {i}, perte : {loss.mean()}")

# plot the results
import matplotlib.pyplot as plt

y_pred = linear.forward(X)
plt.plot(y, label="y")
plt.plot(y_pred, label="y_pred")
plt.legend()
plt.show()
