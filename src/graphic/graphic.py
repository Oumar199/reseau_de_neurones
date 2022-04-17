"""On doit ajouter un graphique pour pouvoir tracer l'évolution des erreurs sur les données d'entraînement et sur les données de test au fur de l'entraînement du réseau
"""


from abc import ABC, abstractmethod
from typing import List, Tuple


class Graphic(ABC):
    """Classe permettant de tracer des graphiques d'évolution des erreurs"""

    @abstractmethod
    def __init__(self, limitations: bool = False) -> None:
        """Initialisation des attributs.

        Args:
            limitations (bool): Indique s'il faut ajouter des limitations sur les axes. Defaults to False.
        """
        ...

    @abstractmethod
    def evolution(
        self,
        train_losses: List,
        test_losses: List,
        epochs: int = 100
    ) -> None:
        """Trace l'évolution des erreurs.

        Args:
            train_losses (List): Liste des erreurs sur les données d'entraînement.
            test_losses (List): Liste des erreurs sur les données de test.
            epochs (int, optional): Le nombre d'itérations. Defaults to 100.
        """
        ...
