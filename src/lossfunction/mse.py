"""La perte quadratique est la moyenne des carrés des différences entre les valeurs réelles
et les valeurs prédites. 

.. math::
    \\frac{1}{m}\sum_{i = 1}^m\left(y_i - f(x_i)\\right)^2
"""
from src.lossfunction.loss import Loss
from src.tenseurs import Tenseur, np


class MSE(Loss):
    """Classe qui permet le calcul de la fonction de perte quadratique.
    Elle hérite de la classe Loss.
    """

    def __init__(self) -> None:
        """Classe qui permet le calcul de la fonction de perte quadratique.
        Elle hérite de la classe Loss.
        """
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
        return (1 / m) * np.nansum((actual - predicted) ** 2)  # type: ignore

    def grad_loss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """Calcul du gradient de la fonction de perte

        Args:
            predicted (Tenseur): Les valeurs prédites
            actual (Tenseur): Les valeurs réelles

        Returns:
            Tenseur: Le gradient de la fonction de perte
        """
        m = actual.shape[1]
        return (2 / m) * (predicted - actual)  # type: ignore
