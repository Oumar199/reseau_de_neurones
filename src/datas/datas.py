"""Nous allons créer un type Batch grâce à la librairie dataclasses. Le batch devra contenir à la fois les features
ou variables explicatives et les données cibles (target).
"""
from dataclasses import make_dataclass
from typing import Iterator

import numpy as np
from src.tenseurs import Tenseur

Batch = make_dataclass("Batch", [("inputs", Tenseur), ("target", Tenseur)])


class DataIterator:
    """Cette classe permet d'itérer sur tous les batchs"""

    def __call__(self, inputs: Tenseur, target: Tenseur) -> Iterator[Batch]:  # type: ignore
        """Appel de la classe permettant d'itérer sur les batchs

        Args:
            inputs (Tenseur): variables explicatives
            target (Tenseur): les données cibles

        Yields:
            Iterator[Batch]: _description_
        """
        pass


class BatchIterator(DataIterator):
    """Cette classe permet l'initialisation et l'itération sur tous les batchs. Elle hérite de la classe
    DataIterator.
    """

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        """Initialisation des paramètres de création de batchs

        Args:
            batch_size (int, optional): Définition de la taille des batchs. Defaults to 32.
            shuffle (bool, optional): Indique si les batchs seront mélangés ou pas. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tenseur, target: Tenseur) -> Iterator[Batch]:  # type: ignore
        """Appel de la classe permettant d'itérer sur les batchs

        Args:
            inputs (Tenseur): variables explicatives
            target (Tenseur): les données cibles

        Yields:
            Iterator[Batch]: _description_
        """
        starts = np.arange(0, inputs.shape[1], self.batch_size)
        for start in starts:
            end = start + self.batch_size
            inputs_batch = inputs[:, start:end]
            target_batch = target[:, start:end]
            yield Batch(inputs_batch, target_batch)
