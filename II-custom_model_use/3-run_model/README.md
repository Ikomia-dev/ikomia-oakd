# Faire tourner un modèle dans l'OAK-D

Pour utiliser un modèle de notre choix, il faut se servir des objets proposés par la [bibliothèque](https://docs.luxonis.com/projects/api/en/latest/references/python/) DepthAI, je vais expliquer pourquoi et comment se servir d'eux.

Pour se faire, j'ai créé plusieurs programmes assez similaire, un par approche possible pour le traitement, elles reposent principalement sur ces 3 objets :
- [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) : approche pour un modèle quelconque.
- [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork) : approche pour un modèle MobileNet.
- [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork) : approche pour un modèle YOLO.
<br><br><br>



## Réseau de neurones quelconque
<br>

### Principe
Pour utiliser un modèle, la façon la plus générique est de passer par l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork), il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée (en se référant à la documentation du modèle).

Gérer la sortie s'avère souvent complexe par rapport aux objets plus spécifiques, cependant, avec une instance NeuralNetwork, il est certain que le modèle pourra être utilisé (en supposant qu'il soit au bon format pour la Myriade X).
<br><br>


### Programme "as_nn_coronamask.py"
Détecte les visages et indique si un masque est porté.

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


### Fonctionnement récupération sortie modèle (détection)
En l'occurence, la sortie correspond à des détections, cependant, il est tout a fait possible d'utiliser la même méthode avec n'importe qu'elle sortie, la seule condition est de pouvoir être interprétée par l'objet [NNData](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NNData).

```py
# Configurer le réseau
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("chemin/vers/le.blob")

# Initier un accès à la sortie
nn_output_stream = pipeline.create(dai.node.XLinkOut)
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer la sortie
detection_current_output = detection_queue.get() # Objet NNData
tensor = detection_current_output.getLayerFp16("DetectionOutput") # Récupère la couche "DetectionOutput"
# Compléter avec une boucle permettant de traiter la sortie
```
<br><br>



## Réseau de neurones [MobileNet](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html)
<br>

### Principe
Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [MobileNet](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html), [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.
<br><br>


### Programme "mdn_coronamask.py"
Pas de notion de profondeur, détecte simplement les visages et indique si un masque est porté.

La détection se fait via l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), la structure de la sortie du modèle est donc celle de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out), un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


### Fonctionnement détection MobileNet
L'instance d'objet ImgDetections contient une liste d'instances [ImgDetection](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetection) sur lesquelles il est très facile d'appliquer un traitement.

```py
# Configurer le réseau
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setBlobPath("chemin/vers/le.blob")

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
<br><br>



## Réseau de neurones [YOLO](https://appsilon.com/object-detection-yolo-algorithm/)
<br>

### Principe
Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [YOLO](https://appsilon.com/object-detection-yolo-algorithm/), [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.
<br><br>


### Programme "ydn_devices.py"
Pas de notion de profondeur, détecte les appareils et les classes.

La détection se fait via l'objet [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork), la structure de la sortie du modèle est donc celle de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out), un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).
<br><br>


### Fonctionnement détection YOLO

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