# Exemple détection du port ou non du masque

Je me suis inspiré de l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask) de Luxonis.


## run_coronamask
Programme sans notion de profondeur, détecte simplement les visages et indique si un masque est porté.

La détection se fait via l'objet générique [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork), il est donc possible de s'inspirer de ce programme pour utiliser n'importe qu'elle modèle de réseau de neurones.

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


## run_coronamask_mobilenet
Programme sans notion de profondeur, détecte simplement les visages et indique si un masque est porté.

Cette fois, la détection se fait via l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), il est plus spécifique que NeuralNetwork et n'est utilisable qu'avec des modèles basés sur [MobileNet-SSD](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html). Ici, c'est un modèle MobileNet-SSD, utiliser l'objet MobileNetDetectionNetwork est possible et va beaucoup simplifier les choses.

Au niveau de la structure de la sortie du modèle, il s'agit donc de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out) qui est un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


## run_coronamask_depth : 
Programme similaire mais avec estimation de la position dans l'espace à l'aide de l'objet générique [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork).

(en cours d'écriture)
<br><br>


## run_coronamask_mobilenet_depth : 
Programme similaire mais avec estimation de la position dans l'espace à l'aide de l'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork).

(en cours d'écriture)