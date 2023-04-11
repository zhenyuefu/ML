import numpy as np
from modules.loss import MSELoss

from modules.linear import Linear


np.random.seed(0)
X = np.linspace(-10, 10, 100)
X = X.reshape(-1, 1)

y = 0.85 * X - 0.72

e = np.random.normal(loc=0, scale=0.5, size=X.shape)

y += e

y = y.reshape(-1, 1)

# Instanciation du module Linear et de la fonction de coût MSELoss
linear = Linear(1, 1, bias=False)
loss_fn = MSELoss()

# Boucle d'apprentissage
for i in range(101):
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
plt.plot(y, "o", label="y")
plt.plot(y_pred, "-", label="y_pred")
plt.legend()
plt.show()
