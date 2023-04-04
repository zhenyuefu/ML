import pickle
from collections import Counter

import numpy as np
from numpy.typing import ArrayLike


def entropy(array: ArrayLike) -> float:
    r"""Compute the entropy of a list of labels.
    the entropy is defined as
    .. math::
        H(Y) = - \sum_{y \in Y} p(y) \log_2 p(y)
    where p(y) is the probability of the label y in the vector.

    Parameters
    ----------
    array : array_like
        a list of labels
    Returns
    -------
    entropy : float
        The calculated entropy.
    """

    occurrence = Counter(array)
    probas = np.array([occurrence[i] / len(array) for i in occurrence])
    return -np.sum(probas * np.log2(probas))


def entropy_conditional(list_array: ArrayLike) -> float:
    r"""Compute the conditional entropy of a list of labels.
    the conditional entropy is defined as
    .. math::
        H(Y|P) = \sum_i p(P_i) H(Y|P_i) \\
        H(Y|P_i) = - \sum_{y \in Y} p(y|P_i) \log_2 p(y|P_i) \\
        p(P_i) = \frac{|P_i|}{\sum_j |P_j|}

    Parameters
    ----------
    list_array : list
        a list of partition of labels
    Returns
    -------
    entropy : float
        The calculated conditional entropy.
    """

    total = sum([len(p) for p in list_array])
    if total == 0:
        return 0
    probas = np.array([len(p) / total for p in list_array])
    return np.sum(probas * np.array([entropy(p) for p in list_array]))


# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre , fields : id feature -> nom
[data, id2titles, fields] = pickle.load(open("imdb_extrait.pkl", "rb"))
# la dernière colonne est le vote
datax = data[:, :32]
datay = np.array([1 if x[33] > 6.5 else -1 for x in data])

votes = datay
# only keep the binary features
features = datax[:, :28]

# Calculer pour chaque attribut binaire l’entropie et l’entropie conditionnelle du vote selon la partition induite
# par l’attribut (les exemples dont la valeur de l’attribut est 1 vs les autres).
ent = entropy(votes)
print(f"Entropy = {ent:.6f}")
differences = []
for i, feature in enumerate(features.T):
    partition = [votes[feature == 1], votes[feature != 1]]
    cond_ent = entropy_conditional(partition)
    field_name = fields[i]
    diff = ent - cond_ent
    differences.append((field_name, diff))
    print(
        f"Attribute {field_name}: Conditional Entropy = {cond_ent:.6f}, difference = {diff:.6f}"
    )

# Afficher l'attribut qui maximise le gain d’information.
print(
    f"Best attribute is {max(differences, key=lambda x: x[1])[0]}, with a difference of {max(differences, key=lambda x: x[1])[1]:.6f}"
)

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTree
import pydotplus

id2genre = [x[1] for x in sorted(fields.items())[: -2]]
dt = DTree()
dt.max_depth = 5  # on fixe la taille max de l’arbre
dt.min_samples_split = 2  # nombre minimum d’exemples pour spliter un nœud
dt.fit(datax, datay)
dt.predict(datax[:5, :])
print(dt.score(datax, datay))
# utiliser http://www.webgraphviz.com/ par exemple ou https://dreampuf.github.io/GraphvizOnline
export_graphviz(dt, out_file="/tmp/tree.dot", feature_names=id2genre)
# ou avec pydotplus
tdot = export_graphviz(dt, feature_names=id2genre)
pydotplus.graph_from_dot_data(tdot).write_pdf('tree.pdf')

