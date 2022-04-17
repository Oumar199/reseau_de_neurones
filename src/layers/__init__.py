"""
Les différentes couches du réseau
----------------------------------------------

Chaque couche d'un réseau doit pouvoir propager les informations vers l'avant (*forward propagation*)
ou vers l'arrière (*backward propagation*). Certaine couche peuvent ainsi utiliser une fonction d'activation.
Les fonctions d'activations sont pour la plupart dites non linéaires.
D'autres couches par contre utilisent des fonctions linéaires. Ces dernières couches ne font qu'additionner
les variables multipliées avec leurs poids (un biais est souvent ajouté au calcul) : 

.. math::
    W\\times x + b


Avec W La matrice des poids, x la matrice des observations et b le biais.
Les différentes fonctions d'activation à utiliser sont les suivantes : *Linéaire* et *Relu*. Il en existe toute une
panoplie mais nous retenons ces fonctions pour l'instant.
"""
