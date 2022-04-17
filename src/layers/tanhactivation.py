"""Fonction d'activation tangente hyperbolique.
"""

import numpy as np
from src.layers.activation import Activation
from src.tenseurs import Tenseur


def tanh(x: Tenseur) -> Tenseur:
    """Fonction d'activation tangente hyperbolique

    Args:
        x (Tenseur): Les sorties de couche linéaire

    Returns:
        Tenseur: Les informations de forward propagation
    """
    return np.tanh(x)


def tanh_prime(x: Tenseur) -> Tenseur:
    """Dérivée de la fonction d'activation tangente hyperbolique

    Args:
        x (Tenseur): Les sorties de couche linéaire stockées

    Returns:
        Tenseur: Les informations de backward propagation renvoyées à la couche linéaire
    """
    return 1 - np.tanh(x) ** 2


class TanhActivation(Activation):
    """Initialisation des attributs

    Args:
        f (F): fonction d'activation
        f_prime (F): dérivée de la fonction d'activation
    """

    def __init__(self, tanh, tanh_prime) -> None:
        super().__init__(tanh, tanh_prime)
