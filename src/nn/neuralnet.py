"""La première couche du réseau accepte en entrées un échantillon qu'on lui fournit. Le réseau va ensuite 
effectuer des transformations dans chaque couche qui le compose (forward propagation) et suivant la 
perte mesurée au niveau de la sortie du réseau, il décidera s'il faut modifier les paramètres avec
la propagation en arrière (backward propagation) et refaire un forward. Ce processus se répète ainsi un 
certain nombre de fois jusqu'à ce que la perte soit la plus minimale possible. Le réseau est configuré 
suivant la séquence de couches qu'on lui fournit et la composition de chaque couche (nombre de noeuds).

**A noter** : Un réseau profond réagira mieux qu'avec un réseau étendu en largeur avec le même nombre de poids (ou paramètres). 
"""

from typing import Iterator, Sequence, Tuple

# from src.layers.lineaire import Lineaire
from src.tenseurs import Tenseur
from src.layers.layer import Layer


class NeuralNetwork:
    """Réseau de neurones"""

    def __init__(self, reseau: Sequence[Layer]) -> None:
        """Initialisation des paramètres

        Args:
            reseau (Sequence[Layer]): Une séquence de couches
        """
        self.reseau = reseau

    def forward(self, inputs: Tenseur) -> Tenseur:
        """Appliquer le forward propagation sur chaque couche

        Args:
            inputs (Tenseur): Les données en entrée du réseau

        Returns:
            Tenseur: La sortie du réseau
        """
        for couche in self.reseau:
            inputs = couche.forward(inputs)
        return inputs

    def backward(self, grad: Tenseur) -> Tenseur:
        """Appliquer le backward propagation sur chaque couche

        Args:
            grad (Tenseur): Le gradient de la dernière couche

        Returns:
            Tenseur: Le dernier gradien calculé lors de la backpropagation
        """
        for couche in reversed(self.reseau):
            grad = couche.backward(grad)
        return grad

    def param_grad(self) -> Iterator[Tuple[Tenseur, Tenseur]]:
        """Récupération des paramètres et des gradients à l'aide d'un générateur

        Yields:
            Iterator[Tuple[Tenseur, Tenseur]]: Les paramètres et gradient du réseau
        """
        for couche in self.reseau:
            for name, param in couche.params.items():
                grad = couche.grads[name]
                yield param, grad
