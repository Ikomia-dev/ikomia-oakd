# Faire tourner le modèle dans le OAK-D

Des exemples d'utilisation sont disponibles dans les repertoires adjacents, ici, je vais expliquer le fonctionnement général.

Pour utiliser un modèle de notre choix, il faut se servir des objets proposés par la bibliothèque DepthAI, je vais expliquer pourquoi et comment se servir de certain d'entre eux.


## NeuralNetwork

Pour utiliser un modèle de notre choix, la façon la plus générique de la faire en en passant par l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) de la bibliothèque DepthAI, il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

```py
# Configurer le réseau
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")

# Initier un accès à la sortie
nn_output_stream = pipeline.createXLinkOut()
nn_output_stream.setStreamName("nn")
nn.out.link(nn_output_stream.input)
detection_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# Récupération de la sortie
detection_current_output = detection_queue.get() # Objet NNData
tensor = detection_current_output.getLayerFp16("DetectionOutput") # Récupère la couche "DetectionOutput"
# Appliquer un traitement approprié
```
<br>


## MobileNetDetectionNetwork

L'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork) est plus spécifique que NeuralNetwork et n'est utilisable qu'avec des modèles basés sur [MobileNet-SSD](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html). Son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique NeuralNetwork.

Au niveau de la structure de la sortie du modèle, il s'agit donc de l'attribut [out](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.DetectionNetwork.out) qui est un objet [ImgDetections](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ImgDetections).

```py
# Configurer le réseau
nn = pipeline.createMobileNetDetectionNetwork()

# Récupération de la sortie
detections = detection_queue.get().detections # Attribut "detections" de l'objet ImgDetections
for detection in detections:
    # Appliquer un traitement approprié
```