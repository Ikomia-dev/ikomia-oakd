# Utilisation


## Depthai

L'OAK-D est utilisable sous les principaux OS, entre autre, il suffit d'avoir un environnement de developpement Python, un compilateur C++, CMake et la [bibliothèque DepthAI](https://docs.luxonis.com/projects/api/en/latest/install/) associée.

```py
pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai
```

Une fois cette bibliothèque installée (avec ses [dépendences](https://github.com/luxonis/depthai-python#installation)), il suffit d'alimenter le OAK-D sur le secteur à hauteur de 5v et le brancher en USB sur la machine possédant l'environnement de developpement. L'alimentation sur le secteur est optionnel si le dispositif est piloté en USB3. Le pilotage peut ensuite se faire via un programme (Python ou C++) utilisant la bibliothèque DepthAI, notamment via l'objet [pipeline](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.Pipeline) qui est le tunnel d'accès vers le OAK-D ([docPython](https://docs.luxonis.com/projects/api/en/latest/references/python/#module-depthai), [docC++](https://docs.luxonis.com/projects/api/en/latest/references/cpp/#c-api-reference)).

Utilisant principalement python, je vais, au début, me concentrer sur l'utilisation du OAK-D via se langage, qui, de toute façon, me semble plus pratique pour la découverte de l'appareil.
<br><br>



## Exemples

Luxonis a ouvert un dépôt [github](https://github.com/luxonis/depthai-python) qui aide à comprendre le fonctionnement de la bibliothèque depthai, il y a notamment des programmes [exemples](https://github.com/luxonis/depthai-python/tree/main/examples) permettant de mieux comprendre le fonctionnement.

Pour ma part, en m'inspirant très fortement de ceux-ci, j'ai créé 2 programmes minimalistes, un pour afficher le flux vidéo de la caméra centrale et l'autre pour évaluer la profondeur (via les caméras lattérales).

Le fonctionnement général est le suivant :

```py
# Initialiser le tunnel d'accès vers le OAK-D
pipeline = dai.Pipeline()

# L'utiliser pour récupérer et configurer une caméra
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(800, 540) # Dimension du flux vidéo récupéré
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB) # Socket à utiliser
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Instancier la sortie du pipeline pour la lier à celle de la caméra
xoutRgb = pipeline.create(dai.node.XLinkOut) # Créé un flux de sortie au pipeline
xoutRgb.setStreamName("rgb") # Nom du flux de sortie associé
camRgb.preview.link(xoutRgb.input) # Lie le flux vidéo et le flux de sortie

# Initier la connexion au OAK-D
with dai.Device(pipeline) as device:
    device.startPipeline() # Démarage de l'appareil
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # Appliquer un traitement à chaque image
    while True:
        inRgb = qRgb.get() # Récupération d'une frame de la fils d'attente
        cv2.imshow("output", inRgb.getCvFrame()) # Affichage de celle-ci
```