"""Cette étape est crucial et fera appel au réseau contenant les couches et qui sera capable d'effectuer une
suite de forward et de backward sur les données. L'optimisation des paramètres sera cruciale pour trouver
les paramètres de prédilection et sera effectuée entre un backward et la reprise du forward.

Certains hyperparamètres seront utiles à l'optimisation du paramètre.
"""
from typing import List

from tqdm import tqdm
from src.graphic.animation import Animation
from src.graphic.graphic import Graphic
from src.lossfunction.logisticloss import LogisticLoss
from src.datas.datas import BatchIterator
from src.tenseurs import Tenseur
from src.lossfunction.mse import MSE
from src.nn.neuralnet import NeuralNetwork
from src.optimisation.sgd import SGD


class Training:
    def __init__(self, lr: float = 0.01, epochs: int = 30) -> None:
        """Initialisation de quelques hyperparamètres d'entraînement. Ces paramètres sont ajustables.

        Args:
            lr (float, optional): Le taux d'apprentissage du réseau. Defaults to 0.01.
            epochs (int, optional): Le nombre d'itération du réseau. Defaults to 30.
        """
        self.lr = lr
        self.epochs = epochs
        self.train_errors: List = []
        self.test_errors: List = []

    def train(self, X_train: Tenseur, X_test: Tenseur, y_train: Tenseur, y_test: Tenseur, batchs: BatchIterator, nn: NeuralNetwork, loss: MSE(), optimiseur: SGD):  # type: ignore
        """Entrainement du réseau

        Args:
            X_train (Tenseur): Les données d'entraînement des variables explicatives
            X_test (Tenseur): Les données de test des variables explicatives
            y_train (Tenseur): Les données d'entraînement de la variable à expliquer
            y_test (Tenseur): Les données de test de la variable à expliquer
            batchs (BatchIterator): Les options de création de batch
            nn (NeuralNetwork): Architecture à adopter
            loss (MSE): La fonction de perte qui sera utilisée
            optimiseur (SGD): La technique d'optimisation à utiliser
        """

        for epoch in tqdm(range(self.epochs)):  # type: ignore
            train_losses = 0.0
            test_predictions = nn.forward(X_test)

            self.test_errors.append(loss.loss(test_predictions, y_test))

            for batch in batchs(X_train, y_train):
                train_predictions = nn.forward(batch.inputs)
                train_losses += loss.loss(train_predictions, batch.target)
                grad = loss.grad_loss(train_predictions, batch.target)
                nn.backward(grad)
                optimiseur(self.lr).step(nn)  # type: ignore

            self.train_errors.append(train_losses)

    def graphic(self, graphic: Graphic) -> None:
        """Affichage de l'évolution des erreurs.

        Args:
            animation (Animation): objet permettant de tracer le graphique d'évolution.
        """
        graphic.evolution(
            self.train_errors,
            self.test_errors,  # type: ignore
            self.epochs,  # type: ignore
        )
