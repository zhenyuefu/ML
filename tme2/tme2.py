import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


class Density(object):
    def fit(self, data):
        pass

    def predict(self, data):
        pass

    def score(self, data):
        # Calcule la log-vraisemblance P(data|model) de l'estimateur pour les données data

        eps = 10e-10
        # Estime la densité de probabilité
        density = np.exp(self.predict(data))
        # Calcule la log-vraisemblance
        log_likelihood = np.sum(np.log(density + eps))

        return log_likelihood


class Histogramme(Density):
    def __init__(self, steps=10):
        Density.__init__(self)
        self.steps = steps

    def fit(self, x):
        # A compléter : apprend l'histogramme de la densité sur x
        self.histo, edges = np.histogramdd(x, bins=self.steps)

    def to_bin(self, data):
        x, y = data[:, 0], data[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        # stepx est le longueur d'un step
        stepx = (xmax - xmin) / self.steps
        stepy = (ymax - ymin) / self.steps
        res = [(int((x - xmin) / stepx), int((y - ymin) / stepy)) for (x, y) in data]
        # res = [(int((d[0]-xmin)/stepx), int((d[1]-ymin)/stepy)) for d in data]
        res = [(xi if xi != self.steps else xi - 1, yi if yi != self.steps else yi - 1) for (xi, yi) in res]
        return res, stepx, stepy

    def predict(self, x):
        # A compléter : retourne la densité associée à chaque point de x
        n = x.shape[0]
        ind, stepx, stepy = self.to_bin(x)
        V = stepx * stepy
        res = []
        for i in ind:
            cx, cy = i
            k = self.histo[cx, cy]
            res.append(k / (n * V))
        return np.array(res)


class KernelDensity(Density):
    def __init__(self, kernel=None, sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self, x):
        self.x = x

    def predict(self, data):
        # Nombre de données d'apprentissage
        n = len(self.x)
        # Nombre de données à prédire
        m = len(data)
        # Calcul des distances euclidiennes entre les données d'apprentissage et les données à prédire
        distances = np.linalg.norm(self.x[:, None] - data[None, :], axis=2)
        # Application du noyau sur les distances normalisées
        kernel_values = self.kernel(distances / self.sigma)
        # Estimation de la densité pour chaque point de data
        density = np.sum(kernel_values, axis=0) / (n * self.sigma ** len(data[0]) * self.kernel(np.zeros(len(data[0]))))
        return density


def kernel_uniform(x):
    # Vérifie si toutes les dimensions sont dans l'intervalle [-0.5, 0.5]
    in_range = np.all(np.abs(x) <= 0.5, axis=1)
    # Initialise le tableau de sortie à zéro
    output = np.zeros(x.shape[0])
    # Affecte la valeur 1 aux exemples qui sont dans l'intervalle [-0.5, 0.5]
    output[in_range] = 1.0
    return output

def kernel_gaussian(x):
    d = x.shape[1]
    return (2*np.pi)**(-d/2) * np.exp(-0.5 * np.linalg.norm(x, axis=1)**2)

def get_density2D(f, data, steps=10):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    xlin, ylin = np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin


def get_density2D(f, data, steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    xlin, ylin = np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin


def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res + 1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi, fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1], v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data, note


plt.ion()
# Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
# La fonction charge la localisation des POIs dans geo_mat et leur note.
geo_mat, notes = load_poi("bar")

# Affiche la carte de Paris
show_img()
# Affiche les POIs
plt.scatter(geo_mat[:, 0], geo_mat[:, 1], alpha=0.8, s=3)

# Affiche la densité estimée
hist = Histogramme(100)
hist.fit(geo_mat)
# res = hist.predict(geo_mat)
show_density(hist, geo_mat, 100)
show_img()

# Affiche la densité estimée avec le noyau gaussien
hist = Histogramme(100, kernel=kernel_gaussian)
hist.fit(geo_mat)
# res = hist.predict(geo_mat)
show_density(hist, geo_mat, 100)
show_img()
