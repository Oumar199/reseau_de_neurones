"""La perte absolue est la moyenne des valeurs absolues des différences entre les valeurs réelles
et les valeurs prédites. 

.. math::
    \\frac{1}{m}\sum_{i = 1}^m\left|y_i - f(x_i)\\right|
"""
from src.lossfunction.loss import Loss
from src.tenseurs import Tenseur, np


class MAE(Loss):
    """Classe qui permet le calcul de la fonction de perte absolue.
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
        return (1 / m) * np.nansum(np.abs(actual - predicted))  # type: ignore

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            Tenseur: Le gradient de la fonction de perte
        """
        m = actual.shape[1]
        epsilon = 1e-10
        return -(1 / m) * ((actual - predicted) / (np.abs(actual - predicted) + epsilon))  # type: ignore
