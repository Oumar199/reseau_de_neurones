"""Couche linéaire : Cette couche utilise une *fonction d'activation linéaire*.
"""

from typing import Dict
from src.initialisation.initialisation import Initialisation
from src.layers.layer import Layer
from src.tenseurs import Tenseur, np


class Lineaire(Layer):
    """Une couche linéaire quelconque du réseau. Elle a pour parent la classe Layer (couche)"""

    def __init__(self, input_size: int, output_size: int, init: Initialisation) -> None:
        """Initialisation des attributs de la couche

        Args:
            input_size (int): La taille des entrées
            output_size (int): La taille des sorties
        """
        super().__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.params["W"] = init.initialisation(self.input_size, self.output_size)
        self.params["b"] = init.initialisation(1, self.output_size)

    def forward(self, inputs: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'avant.

        Args:
            inputs (Tenseur): Les entrées

        Returns:
            Tenseur: Sortie(s) de la couche.
        """
        self.inputs = inputs

        return self.params["W"].dot(inputs) + self.params["b"]

    def backward(self, grad: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'arrière.

        Args:
            grad (Tenseur): Le gradient de la couche.

        Returns:
            Tenseur: l'information qui se propage vers l'arrière.
        """
        self.grads["W"] = grad.dot(self.inputs.T)
        self.grads["b"] = np.nansum(grad, axis=1, keepdims=True)
        return self.params["W"].T.dot(grad)
