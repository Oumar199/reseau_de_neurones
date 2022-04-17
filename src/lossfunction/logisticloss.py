"""Perte logistique. 
"""
from src.lossfunction.loss import Loss
from src.tenseurs import Tenseur, np


class LogisticLoss(Loss):
    """Classe qui permet le calcul de la fonction de perte logistique.
    Elle hérite de la classe Loss.
    """

    def __init__(self) -> None:
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """Implémentation de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            float: Ce que retourne la fonction de perte
        """
        m = actual.shape[1]
        epsilon = 1e-5
        return -(1 / m) * np.nansum(
            actual * np.log(predicted + epsilon)
            + (1 - actual) * np.log(1 - predicted + epsilon)
        )

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            Tenseur: Le gradient de la fonction de perte
        """
        m = actual.shape[1]
        epsilon = 1e-5
        return -(1 / m) * (np.divide(actual, predicted + epsilon) - np.divide((1 - actual), (1 - predicted + epsilon)))  # type: ignore
