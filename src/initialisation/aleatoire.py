"""Initialisation de paramètres avec des valeurs aléatoires.
"""

from src.tenseurs import Tenseur
import numpy as np
from src.initialisation.initialisation import Initialisation


class RandomInit(Initialisation):
    """Classe qui initialise les paramètres avec des valeurs aléatoires"""

    def __init__(self) -> None:
        pass

    def initialisation(self, shape_1: int, shape_2: int) -> Tenseur:
        """Génération des paramètres

        Args:
            shape_1 (int): La taille des entrées
            shape_2 (int): La taille des sorties

        Returns:
            Tenseur: Les paramètres générés.
        """
        return np.random.randn(shape_2, shape_1)  # type: ignore
