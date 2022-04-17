"""Appliquons les tests 
"""


import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import argparse
from typing import List
from src.exemples.exemple_iris import TestIris
from src.exemples.exemple_titanic import TestTitanic
from src.exemples.exemple_boston import TestBoston


def tests(
    sequence: List[str],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    loss: str,
    optimiseur: str,
    initialisation: str,
    data: str,
) -> None:
    """Application des tests suivant les hyperparamètres fournis

    Args:
        sequence (List[str]): Une séquence de couches qui peuvent prendre les valeurs 'lineaire' pour la couche linéaire,
        'sigmoide' pour la couche sigmoide, 'relu' pour la couche relu ou 'tanh' pour la couche tangente hyperbolique. La séquence doit être complète pour
        éviter certains bug.
        batch_size (int): La longueur d'un batch.
        learning_rate (float): Le taux d'apprentissage.
        epochs (int): Le nombre d'itérations.
        loss (str): La fonction perte à utiliser. Peut prendre les valeurs 'logistic' pour la fonction de perte logistique, 'mse'
        pour l'erreur quadratique, 'mae' pour l'erreur absolue ou 'hingeloss' pour la perte hinge.
        optimiseur (str): Le type d'optimiseur à utiliser. Peut prendre la valeur 'sgd' pour la descente de gradient stochastique.
        initialisation (str): Le type d'initialisation des paramètres. Peut prendre les valeurs 'zero' pour l'initialisation avec des zeros,
        'aleatoire' pour l'initialisation avec des nombres aléatoires ou 'he' pour l'initialisation he.
        data (str): Les données qui seront utilisées pour les tests. L'argument peut prendre une valeur entre 'iris' pour le test sur les fleurs iris, 'titanic'
        pour le test sur les données du titanic ou 'boston' pour le test sur les données sur les maisons de boston.
    """
    if data == "iris":
        test_iris = TestIris()
        # effectuons les tests
        test_iris.test(
            sequence,
            batch_size,
            learning_rate,
            epochs,
            loss,
            optimiseur,
            initialisation,
        )
    elif data == "titanic":
        test_titanic = TestTitanic()
        # effectuons les tests
        test_titanic.test(
            sequence,
            batch_size,
            learning_rate,
            epochs,
            loss,
            optimiseur,
            initialisation,
        )
    elif data == "boston":
        test_boston = TestBoston()
        # effectuons les tests
        test_boston.test(
            sequence,
            batch_size,
            learning_rate,
            epochs,
            loss,
            optimiseur,
            initialisation,
        )
    else:
        raise ValueError(
            "Les données {} ne sont pas inclues dans les tests !".format(data)
        )


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-sequence",
        nargs="+",
        help="Une séquence de couches qui peuvent prendre les valeurs 'lineaire' pour la couche linéaire, 'sigmoide' pour la couche sigmoide, 'relu' pour la couche relu ou 'tanh' pour la couche tangente hyperbolique. La séquence doit être complète pour éviter certains bug.",
        type=str,
        default=["lineaire", "sigmoide"],
    )
    parse.add_argument(
        "-batch_size", help="La longueur d'un batch.", type=int, default=32
    )
    parse.add_argument(
        "-learning_rate", help="Le taux d'apprentissage.", type=float, default=0.01
    )
    parse.add_argument("-epochs", help="Le nombre d'itérations.", type=int, default=40)
    parse.add_argument(
        "-loss",
        help="La fonction de perte à utiliser. Peut prendre la valeura 'logistic' pour la fonction de perte logistique, 'mse' pour l'erreur quadratique, 'mae' pour l'erreur absolue ou 'hingeloss' pour la perte hinge.",
        type=str,
        default="logisticloss",
    )
    parse.add_argument(
        "-optimiseur",
        help="le type d'optimiseur à utiliser. Peut prendre la valeur 'sgd' pour la descente de gradient stochastique.",
        type=str,
        default="sgd",
    )
    parse.add_argument(
        "-initialisation",
        help="Le type d'initialisation des paramètres. Peut prendre la valeur 'zero' pour l'initialisation avec des zeros, 'aleatoire' pour l'initialisation avec des nombres aléatoires ou 'he' pour l'initialisation he.",
        type=str,
        default="aleatoire",
    )
    parse.add_argument(
        "-data",
        help="Les données qui seront utilisées pour les tests. L'argument peut prendre une valeur entre 'iris' pour le test sur les fleurs iris, 'titanic' pour le test sur les données du titanic ou 'boston' pour le test sur les données sur les maisons de boston.",
        type=str,
        default="iris",
    )

    args = parse.parse_args()
    tests(args.sequence, args.batch_size, args.learning_rate, args.epochs, args.loss, args.optimiseur, args.initialisation, args.data)  # type: ignore
