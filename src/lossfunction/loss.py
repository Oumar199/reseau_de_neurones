"""Le loss function détermine la qualité de la prédiction. On peut la traduire littéralement par
fonction de perte.
Nous aurons besoin principalement de deux types de loss functions : *MSE*, *LogisticLoss*
On peux en ajouter d'autres comme : *MAE*, *hinge_loss*
"""
from src.tenseurs import Tenseur, np
from abc import ABC, abstractmethod


class Loss(ABC):
    """Classe abstraite qui permet le calcul de la fonction de perte et de
    son gradient.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """Implémentation de la fonction de perte.

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            float: Ce que retourne la fonction de perte
        """
        ...

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            Tenseur: Le gradient de la fonction de perte
        """
        ...
