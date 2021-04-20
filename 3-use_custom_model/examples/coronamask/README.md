# Exemple détection du port ou non du masque

Je me suis inspiré de l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask) de Luxonis.


## run_coronamask
Programme sans notion de profondeur, détecte simplement les visages et indique si un masque est porté.

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


## run_coronamask_depth : 
Programme similaire mais avec estimation de la position dans l'espace.

(en cours d'écriture)