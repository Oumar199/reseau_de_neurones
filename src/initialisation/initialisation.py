"""L'initialisation est une importante très importante car elle peut rendre plus compliquée l'atteinte des paramètres
optimales par le réseau. Les paramètres de chaque couche seront initialisées avec des nombres aléatoires pour certains
et pour d'autres avec des nombres fixes. Des paramètres avec des valeurs d'initialisation très grandes peuvent
souvent prendre plus de temps à être optimisés. 

Il y a principalement trois facons d'initialiser les paramètres:

- Avec des zeros : Toutes les paramètres du réseau seront mis à zero

- De manière aléatoire : Les paramètress sont initialisées de manière aléatoire

- Avec une initialisation HE : Initialisation des paramètres en multipliant les valeurs aléatoires
produit avec un facteur de normalisation

.. math::
    \sqrt{\\frac{2}{\\text{taille des entrées}}}

"""


from abc import ABC, abstractmethod

import numpy as np
from src.tenseurs import Tenseur


class Initialisation(ABC):
    """Classe permettant l'initialisation des paramètres"""

    @abstractmethod
    def __init__(self) -> None:
        """Pas de paramètres d'initialisation."""
        ...

    @abstractmethod
    def initialisation(self, shape_1: int, shape_2: int) -> Tenseur:  # type: ignore
        """Génération des paramètres

        Args:
            shape_1 (int): La taille des entrées
            shape_2 (int): La taille des sorties

        Returns:
            Tenseur: Les paramètres générés.
        """
        ...
