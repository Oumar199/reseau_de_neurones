"""On applique la technique d'optimisation souhaitée.
"""

from src.nn.neuralnet import NeuralNetwork
from src.optimisation.optimize import Optimisation


class SGD(Optimisation):
    """Classe qui permet l'application de la sgd. Elle hérite de la classe Optimisation"""

    def __init__(self, lr: float) -> None:
        """Initialisation du taux d'apprentissage

        Args:
            lr (float): Taux d'apprentissage du réseau
        """
        super().__init__(lr)

    def step(self, reseau: NeuralNetwork):
        """Application de la descente de gradient stochastique

        Args:
            reseau (NeuralNetwork): Le réseau de neurones
        """
        for param, grad in reseau.param_grad():
            param -= self.lr * grad
