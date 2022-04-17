"""Effectuons des tests sur les données sur les maisons de boston
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from src.graphic.animation import Animation
from src.nn.neuralnet import NeuralNetwork
from src.datas.datas import BatchIterator
from src.exemples.test import Test
from src.graphic.simple_graphic import Plot
from src.training.training import Training


class TestBoston(Test):
    """Classe permettant d'effectuer les tests sur les données de boston houses."""

    def __init__(self) -> None:
        pass

    def test(
        self,
        sequence: List[str],
        batch_size: int,
        learning_rate: float,
        epochs: int,
        loss: str,
        optimiseur: str,
        initialisation: str,
    ) -> None:
        """Application des tests suivant les hyperparamètres fournis

        Args:
            sequence (List[str]): Une séquence de couches qui peuvent prendre les valeurs 'lineaire' pour la couche linéaire,
            'sigmoide' pour la couche sigmoide, 'relu' pour la couche relu ou 'tanh' pour tanh. La séquence doit être complète pour
            éviter certains bug.
            batch_size (int): La longueur d'un batch.
            learning_rate (float): Le taux d'apprentissage.
            epochs (int): Le nombre d'itérations.
            loss (str): La fonction perte à utiliser. Peut prendre les valeurs 'logistic' pour la fonction de perte logistique, 'mse'
            pour l'erreur quadratique, 'mae' pour l'erreur absolue ou 'hingeloss' pour la perte hinge.
            optimiseur (str): le type d'optimiseur à utiliser. Peut prendre la valeur 'sgd' pour la descente de gradient stochastique.
            initialisation (str): Le type d'initialisation des paramètres. Peut prendre les valeurs 'zero' pour l'initialisation avec des zeros,
            'aleatoire' pour l'initialisation avec des nombres aléatoires ou 'he' pour l'initialisation he.
        """
        # Données
        boston = load_boston()  # type: ignore

        # Récupération des données
        inputs = boston.data

        # Récupération du target ou de la variable à expliquer
        target = boston.target

        # Normalisation des données
        inputs, normalizer = self.normalisation(inputs)
        
        # Séparation en données d'entraînement et de test
        X_train, X_test, y_train, y_test = self.split_data(inputs, target)  # type: ignore
        X_train, X_test = X_train.T, X_test.T
        y_train, y_test = y_train.reshape(1, y_train.shape[0]), y_test.reshape(
            1, y_test.shape[0]
        )

        # Récupération du type d'initialisation des paramètres
        self.get_initialisation(initialisation)

        # Réseau de neurones
        nn = NeuralNetwork(self.get_sequence(sequence, X_train.shape[0]))  # type: ignore

        # création de batch
        batch = BatchIterator(batch_size)

        # récupération de la fonction de perte
        loss = self.get_loss(loss)  # type: ignore

        # récupération de la technique d'optimisation
        optimisation = self.get_optimiser(optimiseur)  # type: ignore

        # initialisation des options d'entraînement
        Trainer = Training(learning_rate, epochs)

        # Entrainement
        Trainer.train(X_train, X_test, y_train, y_test, batch, nn, loss, optimisation)  # type: ignore

        # création du graphique
        graphic = Plot()

        # affichage des erreurs
        Trainer.graphic(graphic)  # type: ignore

        # print(Trainer.er)
        # Les prédictions finales sur les données detes
        print("---------------------------------------")
        print("Prédiction sur les données de Test :")
        for i, (x, y) in enumerate(zip(X_test.T, y_test.T)):
            x = x.reshape(x.shape[0], 1)
            y = y.reshape(1, y.shape[0])
            predicted = nn.forward(x)
            print("ligne {} : réel -> {} - prédit -> {}".format(i + 1, y, predicted))

        predictions = nn.forward(X_test)
        print("Perte sur les données de test BOSTON : ", loss.loss(predictions, y_test))  # type: ignore