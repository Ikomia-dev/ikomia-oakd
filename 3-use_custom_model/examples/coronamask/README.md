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
Programme avec notion de profondeur, détecte les visages, indique si un masque est porté et donne la position dans l'espace. La position spatiale est obtenue grâce à une instance de l'objet [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator) et la détection se fait via l'objet générique [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork).

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```

Pour utiliser la profondeur, il suffit de passer les zones d'intérêts (ROI) dans la configuration de l'instance de SpatialLocationCalculator, ce qui renvoie une instance d'objet [SpatialLocationCalculatorData](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculatorData) qui permet de récupérer les positions dans l'espace.
<br><br>


## run_coronamask_mobilenet_depth : 
Programme similaire mais avec estimation de la position dans l'espace à l'aide de l'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork), globalement, l'objet est le même que MobileNetDetectionNetwork, sauf qu'ici, en plus du flux vidéo de la caméra rgb, il faut lier la capture de profondeur à l'entrée.

Au niveau de la structure de la sortie du modèle, il s'agit donc de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out) qui est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections)