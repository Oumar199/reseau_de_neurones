"""La perte hinge.
"""
from src.lossfunction.loss import Loss
from src.tenseurs import Tenseur, np


class HingeLoss(Loss):
    """Classe qui permet le calcul de la fonction de perte hinge.
    Elle hérite de la classe Loss.
    """

    def __init__(self) -> None:
        pass

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """Implémentation de la fonction de perte.

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            float: Ce que retourne la fonction de perte
        """
        m = actual.shape[1]
        return (1 / m) * np.nansum(np.maximum(0, 1 - predicted * actual))  # type: ignore

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            Tenseur: Le gradient de la fonction de perte
        """
        m = actual.shape[1]
        return (1 / m) * np.where(predicted * actual < 1, -actual / m, 0)
