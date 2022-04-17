"""Nous devons pouvoir effectuer des tests sur des données. Ainsi il sera nécessaire de devoir fournir des 
hyperparamètres pour effectuer les entraînements et choisir le type de test à effectuer.

Nous effectuerons dans cet exemple un test sur les données concernant les fleurs d'iris.
"""

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.tenseurs import Tenseur
from src.initialisation.aleatoire import RandomInit
from src.initialisation.he_init import HeInit
from src.initialisation.initialisation import Initialisation
from src.initialisation.zeros import ZerosInit
from src.layers.layer import Layer
from src.layers.lineaire import Lineaire
from src.layers.sigactivation import sigmoid, sigmoid_prime, SigActivation
from src.layers.reluactivation import relu, relu_prime, ReluActivation
from src.layers.tanhactivation import tanh, tanh_prime, TanhActivation
from src.lossfunction.logisticloss import LogisticLoss
from src.lossfunction.loss import Loss
from src.lossfunction.mae import MAE
from src.lossfunction.mse import MSE
from src.lossfunction.hinge_loss import HingeLoss
from src.nn.neuralnet import NeuralNetwork
from src.optimisation.optimize import Optimisation
from src.optimisation.sgd import SGD
from src.graphic.animation import Animation


class Test:
    """Classe créée pour effectuer des tests."""

    def __init__(self) -> None:
        """Aucune initialisation."""
        pass

    def get_sequence(self, sequence: List[str], default_input_size: int = 4) -> NeuralNetwork:  # type: ignore
        """Permet de déterminer la séquence de couches

        Args:
            sequence (List[str]): Séquence de couches
        """
        layers: List[Layer] = []
        n_lineaire = 0
        output_size = default_input_size
        for i, couche in enumerate(sequence):
            if couche == "lineaire":
                n_lineaire += 1
                th = "ième" if n_lineaire != 1 else "ière"
                input_size = output_size
                output_size = int(
                    input(
                        "Fournissez la taille des sorties de la {} {} couche linéaire svp : ".format(
                            n_lineaire, th
                        )
                    )
                )
                layers.append(Lineaire(input_size, output_size, self.init))  # type: ignore
            elif couche == "sigmoide":
                layers.append(SigActivation(sigmoid, sigmoid_prime))  # type: ignore
            elif couche == "relu":
                layers.append(ReluActivation(relu, relu_prime))  # type: ignore
            elif couche == "tanh":
                layers.append(TanhActivation(tanh, tanh_prime))  # type: ignore
            else:
                raise TypeError(
                    "La couche {} n'existe pas ! Veuillez vérifier si la valeur fournie est bonne.".format(
                        couche
                    )
                )

        return layers  # type: ignore

    def get_loss(self, loss: str) -> Loss:  # type: ignore
        """Récupération de la fonction de perte

        Args:
            loss (str): Le nom de la fonction de perte

        Returns:
            Loss: La fonction de perte
        """
        if loss == "logistic":
            loss_function = LogisticLoss()  # type: ignore
        elif loss == "mse":
            loss_function = MSE()  # type: ignore
        elif loss == "mae":
            loss_function = MAE()  # type: ignore
        elif loss == "hingeloss":
            loss_function = HingeLoss()  # type: ignore
        else:
            raise TypeError(
                "La fonction {} n'est pas reconnue en tant que fonction de perte. Etes-vous sûr d'avoir choisi la bonne fonction !".format(
                    loss
                )
            )
        return loss_function  # type: ignore

    def get_optimiser(self, optimiseur: str) -> Optimisation:  # type: ignore
        """Récupération de l'optimiseur.

        Args:
            optimiseur (str): Nom de l'optimiseur ou de la technique d'optimisation.

        Returns:
            Optimisation: L'optimiseur.
        """
        if optimiseur == "sgd":
            optimisation = SGD  # type: ignore
        else:
            raise TypeError(
                "La méthode d'optimisation {} n'existe pas !".format(optimiseur)
            )

        return optimisation  # type: ignore

    def get_initialisation(self, initialisation: str) -> Initialisation:  # type: ignore
        """Récupération du type d'initialisation des paramètres.

        Args:
            initialisation (str): Le nom du type d'initialisation.

        Returns:
            Initialisation: La classe d'initialisation.
        """
        if initialisation == "aleatoire":
            self.init = RandomInit()
        elif initialisation == "zeros":
            self.init = ZerosInit()
        elif initialisation == "he":
            self.init = HeInit()
        else:
            raise TypeError(
                "La méthode d'initialisation {} n'existe pas !".format(initialisation)
            )

    def split_data(self, X: Tenseur, y: Tenseur, test_size: float = 0.2, seed: int = 1) -> Tuple[Tenseur]:  # type: ignore
        """Séparation des données en données d'entraînement et en données de test

        Args:
            X (Tenseur): Les variables explicatives
            y (Tenseur): La variable à expliquée
            test_size (float, optional): La taille des données de test. Defaults to 0.2.
            seed (int, optional): _description_. Defaults to 1.

        Returns:
            Tuple[Tenseur]: Les données générées
        """
        if type(X) is pd.DataFrame:
            X, y = X.to_numpy(), y.to_numpy()  # type: ignore
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        return X_train, X_test, y_train, y_test  # type: ignore
    
    def normalisation(self, data: Tenseur, mode: str = "standard")->Tuple[Tenseur, Any]:  # type: ignore
        """Permet la normalisation des données

        Args:
            data (Tenseur): Les données à normaliser
            mode (str, optional): Le mode de normalisation. Peut prendre une valeur entre 'standard'
            pour la standardisation, 'minmax' pour la normalisation min_max ou 'robust' pour la 
            normalisation robuste. Defaults to "standard".

        Returns:
            Tenseur: Les données normalisées
        """
        normalizer = StandardScaler() if mode == "standard" else MinMaxScaler() if mode == "minmax" else RobustScaler()
        
        norm_data = normalizer.fit_transform(data)
        return norm_data, normalizer

    def test(
        self,
        sequence: List[str],
        batch_size: int,
        learning_rate: float,
        epochs: int,
        loss: str,
        optimiseur: str,
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
        """
        ...
