# Mono Neural Inference - Gestion de la profondeur

## Modèle quelconque
<br>

### Principe
Pour obtenir la position dans l'espace 3D, il faut utiliser l'objet [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator) et la détection se fait toujours via l'objet NeuralNetwork.
<br><br>


### Programme "as_nn_coronamask_depth.py" : 
Détecte les visages, indique si un masque est porté et donne la position dans l'espace.

La détection fonctionne toujours pareil, cependant, dans le traitement de la sortie, il y a désormais la récupération des coordonnées spatiales, via l'objet [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator).

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


### Fonctionnement localisation spatiale générique

Pour utiliser la profondeur, il suffit de passer les zones d'intérêts (ROI) rectangulaires dans la configuration de l'instance de SpatialLocationCalculator, ce qui renvoie une instance d'objet [SpatialLocationCalculatorData](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculatorData) qui permet de récupérer les coordonnées dans l'espace 3D.

```py
# Configurer le calculateur de localisation spatiale
spatial_location_calculator = pipeline.create(dai.node.SpatialLocationCalculator)
spatial_location_calculator.setWaitForConfigInput(True)

# Lier le calculateur à une instance d'objet StereoDepth auquelle les caméras lattérales ont été liées.
depth.depth.link(spatial_location_calculator.inputDepth) 

# Créer un lien à l'entrée et à la sortie du calculateur de localisation spatiale
spatial_data_output_stream = pipeline.create(dai.node.XLinkOut)
spatial_data_output_stream.setStreamName("spatialData")
spatial_location_calculator.out.link(spatial_data_output_stream.input)
spatial_config_input_stream = pipeline.create(dai.node.XLinkIn)
spatial_config_input_stream.setStreamName("spatialCalcConfig")
spatial_config_input_stream.out.link(spatial_location_calculator.inputConfig)

# Initier les liens via des fils d'attentes
spatial_config_input_queue = device.getInputQueue("spatialCalcConfig")
spatial_calculator_queue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

# Traitement d'image et définition des zones d'intérêts (ROI)
# ...

# Créer une configuration avec les ROI souhaitées
spatial_calculator_config = dai.SpatialLocationCalculatorConfig()
for x1,y1,x2,y2 in rois:
    # Ajout de la ROI à la future configuration du calculateur
    spatial_config_data = dai.SpatialLocationCalculatorConfigData()
    spatial_config_data.roi = dai.Rect(dai.Point2f(x1, y1), dai.Point2f(x2, y2))
    spatial_calculator_config.addROI(spatial_config_data)

# Récupération des coordonnées spatiales
spatial_config_input_queue.send(spatial_calculator_config) # Envoie de la config
spatial_data = spatial_calculator_queue.get().getSpatialLocations()

# Application d'un traitement à celles-ci
for depth_data in spatial_data:
    print(depth_data.spatialCoordinates.x)
    print(depth_data.spatialCoordinates.y)
    print(depth_data.spatialCoordinates.z)
```
<br><br>



## Modèle spécifique MobileNet
<br>

### Principe
La détection fonctionne pareil qu'avant, sauf qu'il faut maintenant utiliser un objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork), c'est sensiblement la même chose, simplement, maintenant, il faut passer en entrée la profondeur en plus du modèle.
<br><br>


### Programme "as_mdn_coronamask_depth.py" :
Détecte les visages, indique si un masque est porté et donne la position dans l'espace.

Au niveau de la structure de la sortie du modèle, il s'agit toujours de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out), mais cette fois, c'est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections).
<br><br>


### Fonctionnement détection spatiale MobileNet

L'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork) fait implicitement le travail d'un calculateur de localisation spatiale ([SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator)), il nous évite donc la pénible tâche du calcule des localisations spatiales.
```py
# Configurer le réseau de neurones
nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
nn.setConfidenceThreshold(0.5) # Conserve se qui est détecté avec au moins 50% d'assurance
nn.setBlobPath("chemin/vers/le.blob")

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
<br><br>



## Modèle spécifique YOLO
<br>

### Principe
La détection fonctionne pareil qu'avant, sauf qu'il faut maintenant utiliser un objet [YoloSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloSpatialDetectionNetwork), c'est sensiblement la même chose, simplement, maintenant, il faut passer en entrée la profondeur en plus du modèle.
<br><br>


### Programme "as_mdn_coronamask_depth.py" :
Détecte les appareils, les classes et affiche leurs coordonnées spatiales.

Au niveau de la structure de la sortie du modèle, il s'agit toujours de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialDetectionNetwork.out), mais cette fois, c'est un objet [SpatialImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialImgDetections).
<br><br>


### Fonctionnement détection spatiale YOLO

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