# Exemple détection du port ou non du masque

Je me suis inspiré de l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask) de Luxonis.


## run_coronamask
Programme sans notion de profondeur, détecte simplement les visages et indique si un masque est porté.

La détection se fait via l'objet générique [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork).

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


## run_coronamask_mobilenet
Même programme que "run_coronamask" sauf qu'il utilise l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), cela simplifie beaucoup les choses puisque le modèle utilisé est un modèle MobileNet-SSD.

Au niveau de la structure de la sortie du modèle, il s'agit donc de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out) qui est un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


## run_coronamask_depth : 
Programme similaire mais avec estimation de la position dans l'espace à l'aide de l'objet générique [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork).

(en cours d'écriture)
<br><br>


## run_coronamask_mobilenet_depth : 
Programme similaire mais avec estimation de la position dans l'espace à l'aide de l'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork).

(en cours d'écriture)