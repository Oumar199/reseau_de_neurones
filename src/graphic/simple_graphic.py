"""Après l'entraînement du réseau on doit pouvoir tracer l'évolution des erreurs pour vérifier si tout s'est bien passé.
"""

from typing import List, Tuple
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.graphic.graphic import Graphic

sns.set()


class Plot(Graphic):
    """Classe permettant de connaître l'évolution avec un simple graphique."""

    def __init__(self, limitations: bool = False) -> None:
        """Initialisation des attributs.

        Args:
            limitations (bool): Indique s'il faut ajouter des limitations sur les axes. Defaults to False.
        """
        self.limitations = limitations

    def evolution(
        self, train_losses: List, test_losses: List, epochs: int = 100
    ) -> None:
        """Trace l'évolution des erreurs.

        Args:
            train_losses (List): Liste des erreurs sur les données d'entraînement.
            test_losses (List): Liste des erreurs sur les données de test.
            epochs (int, optional): Le nombre d'itérations. Defaults to 100.
        """
        if self.limitations:
            plt.xlim(-10, epochs + 10)
            plt.ylim(
                min(train_losses + test_losses) - 3, max(train_losses + test_losses) + 3
            )
        plt.plot(train_losses, label="train losses")
        plt.plot(test_losses, label="test losses")
        plt.legend()
        plt.show()
