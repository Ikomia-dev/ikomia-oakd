# Fonctionnement OAK-D et DepthAI

Le [OAK-D](https://shop.luxonis.com/collections/all/products/1098obcenclosure "boutique Luxonis : OAK-D") est un produit [Luxonis](https://docs.luxonis.com/en/latest/) combinant carte de programmation et caméras, il vise à mettre à profit le deep learning pour analyser l'environnement en temps réel, il permet de le visualiser en 3D.
<br><br>


## Hardware

Le OAK-D est un dérivé du [LUX-D](https://drive.google.com/file/d/1g0bQDLNnpVC_1-AGaPmC8BaXtGaNNdTR/view "fiche technique LUX-D"), il sagit simplement de sa version avec un support physique associé (une protection autour de la carte).

L'appareil possède 3 caméras, une centrale filmant jusqu'en 4K avec un taux de rafraichissement de 60Hz et deux lattérales, filmant jusqu'en 720p avec un taux de rafraichissement de 120Hz. L'idée, c'est que la caméra centrale sert à capturer le flux vidéo, tandis que les caméras lattérales servent à capturer la profondeur de la scène (grâce au croisement de leurs flux)

Au niveau du processeur, celui utilisé est l'[Intel® Movidius™ Myriad™ X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html), d'après la [fiche technique](https://drive.google.com/file/d/1z7QiCn6SF3Yx977oH41Kcq68Ay6e9h3_/view "fiche technique modèles de DepthAI"), il s'agit plus précisément du [MA2485](https://ark.intel.com/content/www/us/en/ark/products/125926/intel-movidius-myriad-x-vision-processing-unit-4gb.html) ou du [MA2085](https://ark.intel.com/content/www/us/en/ark/products/204770/intel-movidius-myriad-x-vision-processing-unit-0gb.html).
<br><br>



## Utilisation

Le OAK-D est utilisable sous les principaux OS, entre autre, il suffit d'avoir un environnement de developpement C++ ou Python et de télécharger la [bibliothèque DepthAI](https://docs.luxonis.com/projects/api/en/latest/install/) associée.
```py
pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai
```

Une fois cette bibliothèque installée (avec ses [dépendences](https://github.com/luxonis/depthai-python#installation)), il suffit d'alimenter le OAK-D sur le secteur (5v) et le brancher en USB sur la machine possédant l'environnement de developpement. Le pilotage peut ensuite se faire via un programme (Python ou C++) utilisant la bibliothèque DepthAI, notamment via l'objet [pipeline](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.Pipeline) qui est le tunnel d'accès vers le OAK-D ([docPython](https://docs.luxonis.com/projects/api/en/latest/references/python/#module-depthai), [docC++](https://docs.luxonis.com/projects/api/en/latest/references/cpp/#c-api-reference)).

Utilisant principalement python, je vais, au début, me concentrer sur l'utilisation du OAK-D via se langage, qui, de toute façon, me semble plus pratique pour la découverte de l'appareil.
<br><br>



## Programmes exemples

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
