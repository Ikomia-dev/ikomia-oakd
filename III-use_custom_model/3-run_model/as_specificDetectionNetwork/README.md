# Utilisation d'un modèle de détection spécifique  ([MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork))


Certains objets sont très spécifiques, il en existe beaucoup, le fonctionnement est souvent le même, je ne vais donc en présenter qu'un, celui de l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork).

Comme son nom l'indique, l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork) est plus spécifique que NeuralNetwork et DetectionNetwork, il n'est utilisable qu'avec des modèles basés sur [MobileNet-SSD](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html). Son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport aux objets plus génériques.
<br><br>


## Programme "sdn_coronamask.py"
Pas de notion de profondeur, détecte simplement les visages et indique si un masque est porté.

La détection se fait via l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), la structure de la sortie du modèle est donc celle de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out), un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


## Fonctionnement détection (MobileNet)

L'instance d'objet ImgDetections contient une liste d'instances [ImgDetection](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetection) sur lesquelles il est très facile d'appliquer un traitement.

```py
# Configurer le réseau
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath("chemin/vers/le.blob")

# Initier un accès à la sortie
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer la sortie
detections = detection_queue.get().detections # Liste des detections
for detection in detections:
    # Appliquer un traitement approprié sur chaque detection
```
<br>


## Programme "sdn_coronamask_depth.py"
A une notion de profondeur, détecte les visages, indique si un masque est porté et donne la position dans l'espace.

Au niveau de la structure de la sortie du modèle, il s'agit toujours de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out), mais cette fois, c'est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections).
<br><br>


## Fonctionnement détection spatiale

(à venir)