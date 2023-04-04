from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


def mse(w, x, y):
    y_pred = np.dot(x, w)
    mse = np.mean((y_pred - y) ** 2)
    return mse


def mse_grad(w, x, y):
    y_pred = np.dot(x, w)
    mse_gradient = np.dot(x.T, (y_pred - y)) * 2 / x.shape[0]
    return mse_gradient


def reglog(w, x, y):
    z = np.dot(x, w)
    cost = np.mean(np.log(1 + np.exp(-y * z)))
    return cost


def reglog_grad(w, x, y):
    z = np.dot(x, w)
    return np.dot(x.T, -y * np.exp(-y * z) / (1 + np.exp(-y * z))) / x.shape[0]


def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1], 1)
    assert np.isclose(mse(wrandom, datax, datay).mean(), 0.54731, rtol=1e-4)
    assert np.isclose(reglog(wrandom, datax, datay).mean(), 0.57053, rtol=1e-4)
    assert np.isclose(mse_grad(wrandom, datax, datay).mean(), -1.43120, rtol=1e-4)
    assert np.isclose(reglog_grad(wrandom, datax, datay).mean(), -0.42714, rtol=1e-4)
    np.random.seed()


def descente_gradient(datax, datay, f_loss, f_grad, eps, n_iter):
    w = np.zeros((datax.shape[1], 1))
    w_history = [w]
    loss_history = [f_loss(w, datax, datay)]
    for _ in range(n_iter):
        grad = f_grad(w, datax, datay)
        w = w - eps * grad
        w_history.append(w)
        loss_history.append(f_loss(w, datax, datay))
    return w, w_history, loss_history


def plot_cost_and_trajectory(datax, datay, f_loss, f_grad, w_history, title):
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(
        x_grid,
        y_grid,
        np.array([f_loss(w.reshape(-1, 1), datax, datay) for w in grid]).reshape(
            x_grid.shape
        ),
        levels=20,
    )
    plt.colorbar()
    plt.title(title)

    ## Ajout de la trajectoire de l'algorithme d'optimisation
    w_history = np.array(w_history).reshape(-1, datax.shape[1])
    plt.plot(w_history[:, 0], w_history[:, 1], "w-o", markersize=2)

    plt.show()


if __name__ == "__main__":
    check_fonctions()
    # Génération des données
    datax, datay = gen_arti(data_type=0, epsilon=0.1, nbex=1600)

    # split train/test
    trainx, trainy = datax[:800], datay[:800]
    testx, testy = datax[800:], datay[800:]

    ## Descente de gradient pour la régression linéaire
    w_lin, w_history_lin, cost_history_lin = descente_gradient(
        trainx, trainy, mse, mse_grad, 0.1, 1000
    )
    plot_frontiere(datax, lambda x: np.sign(x.dot(w_lin)), step=100)
    plot_data(datax, datay)
    plt.show()

    ## Descente de gradient pour la régression logistique
    w_log, w_history_log, cost_history_log = descente_gradient(
        trainx, trainy, reglog, reglog_grad, 0.1, 1000
    )
    plot_frontiere(datax, lambda x: np.sign(x.dot(w_log)), step=100)
    plot_data(datax, datay)
    plt.show()
    # Régression linéaire
    testy_pred_linear = np.sign(np.dot(testx, w_lin))
    accuracy_linear = np.mean(testy_pred_linear == testy)

    # Régression logistique
    testy_pred_logistic = np.sign(np.dot(testx, w_log))
    accuracy_logistic = np.mean(testy_pred_logistic == testy)

    print("Précision de la régression linéaire :", accuracy_linear)
    print("Précision de la régression logistique :", accuracy_logistic)

    ## Visualisation de la fonction de coût et de la trajectoire de l'algorithme d'optimisation
    plot_cost_and_trajectory(
        trainx, trainy, mse, mse_grad, w_history_lin, "MSE (Régression linéaire)"
    )
    plot_cost_and_trajectory(
        trainx, trainy, reglog, reglog_grad, w_history_log, "Régression logistique"
    )
