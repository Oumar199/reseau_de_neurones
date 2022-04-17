"""La couche d'activation est la couche qui vient juste après la couche linéaire.
Ainsi, elle va prendre en entrées les sorties de la couche linéaire lors de la propagation
en avant et la couche linéaire va prendre le gradient calculé au niveau de la couche
d'activation lors de la propagation en arrière.
"""
from turtle import forward
from typing import Callable, Dict
from src.layers.layer import Layer
from src.tenseurs import Tenseur


F = Callable[[Tenseur], Tenseur]


class Activation(Layer):
    """Couche d'activation quelconque"""

    def __init__(self, f: F, f_prime: F) -> None:
        """Initialisation des attributs

        Args:
            f (F): fonction d'activation
            f_prime (F): dérivée de la fonction d'activation
        """
        self.params: Dict[str, Tenseur] = {}
        self.grads: Dict[str, Tenseur] = {}
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'avant.

        Args:
            inputs (Tenseur): Les entrées

        Returns:
            Tenseur: Sortie(s) de la couche.
        """
        self.inputs = inputs
        return self.f(self.inputs)

    def backward(self, grad: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'arrière.

        Args:
            grad (Tenseur): Le gradient de la couche.

        Returns:
            Tenseur: l'information qui se propage vers l'arrière.
        """
        return self.f_prime(self.inputs) * grad
