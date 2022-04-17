"""Fonction d'activation sigmoid.
"""
import numpy as np
from src.layers.activation import Activation
from src.tenseurs import Tenseur


def sigmoid(x: Tenseur) -> Tenseur:
    """Fonction d'activation sigmoid

    Args:
        x (Tenseur): Les sorties de couche linéaire

    Returns:
        Tenseur: Les informations de forward propagation
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tenseur) -> Tenseur:
    """Dérivée de la fonction d'activation sigmoid

    Args:
        x (Tenseur): Les sorties de couche linéaire stockées

    Returns:
        Tenseur: Les informations de backward propagation renvoyées à la couche linéaire
    """
    return sigmoid(x) * (1 - sigmoid(x))


class SigActivation(Activation):
    """Classe de la couche d'activation sigmoid"""

    def __init__(self, sigmoid, sigmoid_prime) -> None:
        """Initialisation des attributs

        Args:
            f (F): fonction d'activation
            f_prime (F): dérivée de la fonction d'activation
        """
        super().__init__(sigmoid, sigmoid_prime)
