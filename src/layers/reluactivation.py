"""Fonction d'activation relu.
"""

import numpy as np
from src.layers.activation import Activation
from src.tenseurs import Tenseur


def relu(x: Tenseur) -> Tenseur:
    """Fonction d'activation relu

    Args:
        x (Tenseur): Les sorties de couche linéaire

    Returns:
        Tenseur: Les informations de forward propagation
    """
    return np.maximum(0, x)


def relu_prime(x: Tenseur) -> Tenseur:
    """_summary_

    Args:
        x (Tenseur): _description_

    Returns:
        Tenseur: _description_
    """
    return np.where(x > 0, 1, 0)


class ReluActivation(Activation):
    """Couche d'activation relu"""

    def __init__(self, relu, relu_prime) -> None:
        """Initialisation des attributs

        Args:
            f (F): fonction d'activation
            f_prime (F): dérivée de la fonction d'activation
        """
        super().__init__(relu, relu_prime)
