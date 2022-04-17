"""La propagation des informations entre l'entrée et la sortie passe par les couches du réseau.
Il y a deux types de propagations : 

- **forward propagation** ou propagation en avant ;

- **backward propagation** ou propagation en arrière.
"""

from typing import Dict
from src.tenseurs import Tenseur, np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Une couche quelconque du réseau. C'est une classe abstraites."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialisation des attributs de la couche

        Args:
            input_size (int): La taille des entrées
            output_size (int): La taille des sorties
        """
        self.params: Dict[str, Tenseur] = {}
        self.grads: Dict[str, Tenseur] = {}

    @abstractmethod
    def forward(self, inputs: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'avant.

        Args:
            inputs (Tenseur): Les entrées

        Returns:
            Tenseur: Sortie(s) de la couche.
        """
        ...

    @abstractmethod
    def backward(self, grad: Tenseur) -> Tenseur:
        """Fonction de propagation de l'information vers l'arrière.

        Args:
            grad (Tenseur): Le gradient de la couche.

        Returns:
            Tenseur: l'information qui se propage vers l'arrière.
        """
        ...
