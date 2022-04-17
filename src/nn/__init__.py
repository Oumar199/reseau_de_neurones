"""
Réseau de neurones
----------------------------------------------

Le réseau comprend plusieurs couches. Chaque couche applique une certaine transformation à ces entrées 
puis renvoie le résultat de cette transformation. La dernière couche du réseau va renvoyer la sortie finale
suivant le type d'apprentissage. La dernière couche comporte une fonction d'activation *sigmoid* s'il s'agit
d'une classification binaire, *softmax* s'il s'agît d'une classification multi-classe et *fonction linéaire* s'il
s'agit d'une régression. 

Suivant la prédiction effectuée, on saura ainsi s'il faut ou pas corriger les paramètres du réseau.
"""
