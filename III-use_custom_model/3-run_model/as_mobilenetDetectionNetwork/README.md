# Utilisation d'un modèle de détection MobileNet


Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [MobileNet](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html), [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.
<br><br>


## Programme "mdn_coronamask.py"
Pas de notion de profondeur, détecte simplement les visages et indique si un masque est porté.

La détection se fait via l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), la structure de la sortie du modèle est donc celle de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out), un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


## Fonctionnement détection MobileNet

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


## Programme "mdn_coronamask_depth.py"
A une notion de profondeur, détecte les visages, indique si un masque est porté et donne la position dans l'espace.

La détection fonctionne pareil qu'avant, sauf qu'il faut maintenant utiliser un objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork), c'est sensiblement la même chose, simplement, maintenant, il faut passer en entrée la profondeur en plus du modèle.

Au niveau de la structure de la sortie du modèle, il s'agit toujours de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out), mais cette fois, c'est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections).
<br><br>


## Fonctionnement détection spatiale MobileNet

L'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork) fait implicitement le travail d'un calculateur de localisation spatiale ([SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator)), il nous évite donc la pénible tâche du calcule des localisations spatiales.
```py
# Configurer le réseau de neurones
nn = pipeline.createMobileNetSpatialDetectionNetwork()
nn.setConfidenceThreshold(0.5) # Conserve se qui est détecté avec au moins 50% d'assurance
nn.setBlobPath("chemin/vers/le.blob")

# Lier le réseau à une instance d'objet StereoDepth auquelle les caméras lattérales ont été liées.
depth.depth.link(nn.inputDepth)

# Initier un accès à la sortie
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer les détections
detections = detection_queue.get().detections
# Appliquer un traitement approprié
```