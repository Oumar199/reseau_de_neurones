"""Initialisation HE.
"""

from src.tenseurs import Tenseur
import numpy as np
from src.initialisation.initialisation import Initialisation


class HeInit(Initialisation):
    """Classe qui initialise les paramètres avec l'initialisation HE"""

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
        return np.random.randn(shape_2, shape_1) * np.sqrt(1 / shape_1)  # type: ignore
