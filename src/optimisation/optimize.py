"""La technique d'optimisation utilisée lors de l'optimisation des paramètres du réseau sera 
la descente de gradient stochastique. Elle permet de diminuer ou d'augmenter les paramètres du réseau
suivant un sous-échantillon choisit aléatoirement.
"""

from src.nn.neuralnet import NeuralNetwork


class Optimisation:
    """Optimisation des paramètres du réseau."""

    def __init__(self, lr: float) -> None:
        """Initialisation du taux d'apprentissage

        Args:
            lr (float): Taux d'apprentissage du réseau
        """
        self.lr = lr

    def step(self, reseau: NeuralNetwork):
        """Application de la descente de gradient stochastique

        Args:
            reseau (NeuralNetwork): Le réseau de neurones
        """
        ...
