# Utilisation d'un modèle de manière générique ([NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork))

L'instance d'un objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) est la plus générique possible pour utiliser un modèle. Son utilisation est donc plus complexe qu'avec des objets plus spécifiques, cependant, cela permet d'utiliser n'importe qu'elle réseau de neurones (en supposant qu'il soit au bon format pour la Myriade X).

Pour obtenir la position dans l'espace 3D, il faut utiliser l'objet [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator) et la détection se fait toujours via l'objet NeuralNetwork.
<br><br>


## Programme "nn_coronamask.py"
Pas de notion de profondeur, détecte simplement les visages et indique si un masque est porté.

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


## Fonctionnement récupération sortie modèle (détection)
En l'occurence, la sortie correspond à des détections, cependant, il est tout a fait possible d'utiliser la même méthode avec n'importe qu'elle sortie, la seule condition est de pouvoir être interprétée par l'objet [NNData](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NNData).

```py
# Configurer le réseau
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")

# Initier un accès à la sortie
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupérer la sortie
detection_current_output = detection_queue.get() # Objet NNData
tensor = detection_current_output.getLayerFp16("DetectionOutput") # Récupère la couche "DetectionOutput"
# Compléter avec une boucle permettant de traiter la sortie
```
<br>


## Programme "nn_coronamask_depth.py" : 
A une notion de profondeur, détecte les visages, indique si un masque est porté et donne la position dans l'espace.


La détection fonctionne toujours pareil, cependant, dans le traitement de la sortie, il y a désormais la récupération des coordonnées spatiales, via l'objet [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator).

Structure de la sortie du modèle :
```json
Liste de 100 zones d'intérêts potentielles, suivant cette structure :
["id", "label", "confidence", "left", "top", "right", "bottom"]
```
<br>


## Fonctionnement localisation spatiale générique

Pour utiliser la profondeur, il suffit de passer les zones d'intérêts (ROI) dans la configuration de l'instance de SpatialLocationCalculator, ce qui renvoie une instance d'objet [SpatialLocationCalculatorData](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculatorData) qui permet de récupérer les coordonnées dans l'espace 3D.

```py
# Configurer le calculateur de localisation spatiale
spatial_location_calculator = pipeline.createSpatialLocationCalculator()
spatial_location_calculator.setWaitForConfigInput(True)

# Lier le calculateur à une instance d'objet StereoDepth auquelle les caméras lattérales ont été liées.
depth.depth.link(spatial_location_calculator.inputDepth) 

# Créer un lien à l'entrée et à la sortie du calculateur de localisation spatiale
spatial_data_output_stream = pipeline.createXLinkOut()
spatial_data_output_stream.setStreamName("spatialData")
spatial_location_calculator.out.link(spatial_data_output_stream.input)
spatial_config_input_stream = pipeline.createXLinkIn()
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