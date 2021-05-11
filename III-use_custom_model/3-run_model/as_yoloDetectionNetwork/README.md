# Utilisation d'un modèle de détection YOLO


Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [YOLO](https://appsilon.com/object-detection-yolo-algorithm/), [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.
<br><br>


## Programme "ydn_devices.py"
Pas de notion de profondeur, détecte les appareils et les classes.

La détection se fait via l'objet [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork), la structure de la sortie du modèle est donc celle de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out), un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


## Fonctionnement détection YOLO

L'instance d'objet ImgDetections contient une liste d'instances [ImgDetection](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetection) sur lesquelles il est très facile d'appliquer un traitement.

```py
# Configurer le réseau
nn = pipeline.create(dai.node.YoloDetectionNetwork)
nn.setConfidenceThreshold(0.5) # Conserve se qui est détecté avec au moins 50% d'assurance
nn.setBlobPath("chemin/vers/le.blob")
nn.setNumClasses(80) # 80 labels différents
nn.setCoordinateSize(4)
nn.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
nn.setIouThreshold(0.5)

# Initier un accès à la sortie
nn_output_stream = pipeline.create(dai.node.XLinkOut)
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer la sortie
detections = detection_queue.get().detections # Liste des detections
for detection in detections:
    # Appliquer un traitement approprié sur chaque detection
```
<br>


## Programme "ydn_devices_depth.py"
A une notion de profondeur, détecte les appareils, les classes et affiche leurs coordonnées spatiales.

La détection fonctionne pareil qu'avant, sauf qu'il faut maintenant utiliser un objet [YoloSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloSpatialDetectionNetwork), c'est sensiblement la même chose, simplement, maintenant, il faut passer en entrée la profondeur en plus du modèle.

Au niveau de la structure de la sortie du modèle, il s'agit toujours de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out), mais cette fois, c'est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections).
<br><br>


## Fonctionnement détection spatiale YOLO

L'objet [YoloSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloSpatialDetectionNetwork) fait implicitement le travail d'un calculateur de localisation spatiale ([SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator)), il nous évite donc la pénible tâche du calcule des localisations spatiales.
```py
# Configurer le réseau de neurones
nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
nn.setConfidenceThreshold(0.5) # Conserve se qui est détecté avec au moins 50% d'assurance
nn.setBlobPath("chemin/vers/le.blob")
nn.setNumClasses(80) # 80 labels différents
nn.setCoordinateSize(4)
nn.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
nn.setIouThreshold(0.5)

# Lier le réseau à une instance d'objet StereoDepth auquelle les caméras lattérales ont été liées.
depth.depth.link(nn.inputDepth)

# Initier un accès à la sortie
nn_output_stream = pipeline.create(dai.node.XLinkOut)
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer les détections
detections = detection_queue.get().detections
# Appliquer un traitement approprié
```