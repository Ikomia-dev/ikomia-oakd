# Mode [stereo neural inference](https://docs.luxonis.com/en/latest/pages/faq/#id2)

L'idée est de faire tourner un modèle en prenant en entrée les flux vidéos des caméras latérales, pour calculer précisément la profondeur en croisant leurs flux. Je me suis aidé d'un [exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation) de Luxonis (presque l'unique exemple utilisant l'inférence neuronale stéréo).

Pour ce faire, il faut s'assurer que chacune des caméras repère les mêmes points, sinon, croiser leurs flux n'aurait pas de sens. C'est pourquoi, les programmes qui vont suivre s'appuieront sur l'identification de points d'intérêts du visage. En effet, admettons qu'il y ait une seule personne dans le champ visuel, si sur chacune des caméras un nez est repéré, alors, il est possible de récupérer ses coordonnées spatiales en croisant les flux.
<br><br>


## face_detector
Programme permettant de détecter un visage (celui avec le plus de confiance), à partir des flux vidéos des caméras latérales.

Globalement, cela fonctionne un peu comme lors de l'utilisation d'une seule caméra, c'est simplement qu'il faut répéter les opérations autant de fois qu'il y en a. En l'occurrence, pour chaque caméra (gauche puis droite), il faut l'initialiser, convertir son flux vidéo, récupérer sa sortie et y lier le réseau de neurones souhaité (détection de visage, avec ce [modèle](https://github.com/luxonis/depthai-experiments/blob/master/gen2-triangulation/models/face-detection-retail-0004_2021.3_6shaves.blob)).

Convertir le flux vidéo des caméras latérales est obligatoire, celles-ci filment sans couleur,
or, le modèle utilisé prend en entrée une image colorée, il faut donc dupliquer la valeur de l'unique pixel pour obtenir une image en BGR.
```py
for side in ["left", "right"]:
    # Initialise la caméra
    cam = pipeline.createMonoCamera()
    if(side == "left"):
        cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # Transforme le flux vidéo (en nuances de gris) en une sortie appropriée (BGR)
    face_manip = pipeline.createImageManip()
    face_manip.initialConfig.setResize(300, 300)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(face_manip.inputImage)

    # Configure le flux de sortie de la caméra
    face_manip_output_stream = pipeline.createXLinkOut()
    face_manip_output_stream.setStreamName(side + "_cam")
    face_manip.out.link(face_manip_output_stream.input)

    # Initialise le réseau de neurones de détection de visage
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath("/chemin/vers/le.blob")
    face_manip.out.link(face_nn.input)

    # Configure le flux de sortie du réseau
    face_nn_output_stream = pipeline.createXLinkOut()
    face_nn_output_stream.setStreamName("nn_" + side + "_faces")
    face_nn.out.link(face_nn_output_stream.input)
```

Ensuite, il suffit d'initialiser les fils de sorties et appliquer un traitement, ici, dessine un rectangle autour de la zone d'intérêt (visage).
```py
with dai.Device(pipeline) as device:
    device.startPipeline()
    output_queues = dict()
    for side in ["left", "right"]:
        output_queues[side+"_cam"] = device.getOutputQueue(name=side+"_cam")
        output_queues["nn_"+side+"_faces"] = device.getOutputQueue(name="nn_"+side+"_faces")
    
    while True:
        for side in ["left", "right"]:
            frame = output_queues[side+"_cam"].get().getCvFrame()
            faces_data = output_queues["nn_"+side+"_faces"].get().getFirstLayerFp16()
            if(faces_data[2] > 0.2): # récupère la détection la plus sure (si > 20%)
                drawROI(frame, (faces_data[3],faces_data[4]), (faces_data[5],faces_data[6]))
            cv2.imshow(side, frame)
```
<br><br>


## face_landmarks_detector
Programme similaire, mais qui, à l'intérieur de la zone du visage, va identifier 5 points d'intérêts (yeux, nez, bouche).

Le fonctionnement ne change donc pas énormément, pour chaque caméra, il faut simplement initialiser un nouveau [modèle](https://github.com/luxonis/depthai-experiments/blob/master/gen2-triangulation/models/landmarks-regression-retail-0009_2021.3_6shaves.blob) (détection de points d'intérêts). Ensuite, lors du traitement, il faut redimensionner la zone du visage détecté et l'insérer en entrée du nouveau réseau de neurones, puis appliquer un nouveau traitement (dessiner les points d'intérêts).

```py
for side in ["left", "right"]:
    # ...

    # Initialise le réseau de neurones de détection de points d'intérêts
    landmarks_nn = pipeline.createNeuralNetwork()
    landmarks_nn.setBlobPath("/chemin/vers/le.blob")

    # Configure le flux d'entrée du réseau
    landmarks_nn_input_stream = pipeline.createXLinkIn()
    landmarks_nn_input_stream.setStreamName("nn_" + side + "_landmarks")
    landmarks_nn_input_stream.out.link(landmarks_nn.input)

    # Configure le flux de sortie du réseau
    landmarks_nn_output_stream = pipeline.createXLinkOut()
    landmarks_nn_output_stream.setStreamName("nn_" + side + "_landmarks")
    landmarks_nn.out.link(landmarks_nn_output_stream.input)


from utils.compute import to_planar # Passe un tableau 2D en 1D (formatage necessaire au nn)

with dai.Device(pipeline) as device:
    # ...
    while True:
        for side in ["left", "right"]:
            # ...
            xmin, ymin = faces_data[3], faces_data[4]
            xmax, ymax = faces_data[5], faces_data[6]
            land_data = dai.NNData()
            planar_cropped_face = to_planar(frame[ymin:ymax, xmin:xmax], (48,48))
            land_data.setLayer("0", planar_cropped_face) # "0" est la couche d'entrée
            input_queues["nn_"+side+"_landmarks"].send(land_data)

            output = output_queues["nn_"+side+"_landmarks"].get().getFirstLayerFp16()
            landmarks = np.array(output).reshape(5,2)

            for x,y in landmarks:
                cv2.circle(frame, (int(x*(xmax-xmin))+xmin,int(y*(ymax-ymin))+ymin), 2, (0,0,255))
            cv2.imshow(side, frame)
```
<br><br>


## face_visualizer
Programme similaire, mais qui, à partir des points d'intérêts détectés, calcule les coordonnées spatiales et visualise les points d'intérêts en 3D (grâce à [pygame](https://www.pygame.org/news) et [OpenGL](https://www.opengl.org//)). Le fichier "visualize.py" regroupe des classes pour visualiser des données en 3D.
- LandmarksCubeVisualizer : Points d'intérêts centrés avec tracé de la zone d'intérêt (pour observer la forme).
- LandmarksDepthVisualizer : Points d'intérêts placés dans le plan par les intersections des vecteurs entre caméras et points (pour observer la profondeur).

```py
for side in ["left", "right"]:
    # ...


cams = dict() # Coordonnées spatiales des caméras (du OAK-D)
cams["left"] = (0.107, -0.038, 0.008)
cams["right"] = (0.109, 0.039, 0.008)

colors = [(255,255,255), (255,255,255), (0,255,255), (255,0,255), (255,0,255)]
pairs = [(0,2), (1,2), (3,4)] # lie les yeux au nez, lie les points de la bouche entre eux

from utils.visualize import LandmarksCubeVisualizer # LandmarksDepthVisualizer (autre point de vue)
visualizer = LandmarksCubeVisualizer(300, 300, [cams["left"], cams["right"]], colors, pairs)
```

Les détails du calcul de profondeur peuvent être assez complexes, je ne vais donc qu'expliquer le principe. Grâce aux points d'intérêts détectés (qui sont à priori les mêmes sur chaque caméra), il est possible de déterminer un vecteur allant d'une caméra à un point d'intérêt, ensuite, en croisant ces vecteurs, cela donne les coordonnées de chaque point d'intérêt.

Tout ceci est de la [stereo vision](https://en.wikipedia.org/wiki/Computer_stereo_vision) et s'inspire de la [parallaxe](https://fr.wikipedia.org/wiki/Parallaxe) (en astronomie).

```py
from utils.compute import to_planar, get_landmark_3d, get_vector_intersection

with dai.Device(pipeline) as device:
    spatial_vectors = dict()
    # ...
    for side in ["left", "right"]:
        spatial_vectors[side] = []
        # ...
        # Détermine les vecteurs spatiales (de chaque caméra à chaque point d'intérêt)
        spatial_landmarks = [get_landmark_3d((x,y)) for x,y in landmarks]
        for i in range(5):
            spatial_vectors[side].append([spatial_landmarks[i][j] - cams[side][j] for j in range(3)])
        spatial_vectors[side] = np.array(spatial_vectors[side]) # conversion : liste -> numpy array
        cv2.imshow(side, frame)

    # Détermine la profondeur pour localiser précisemment les points dans l'espace
    landmark_spatial_locations = []
    if(len(spatial_vectors["left"])>4 and len(spatial_vectors["right"])>4):
        for i in range(5):
            landmark_spatial_locations.append(
                get_vector_intersection(
                    spatial_vectors["left"][i], camera_locations["left"], 
                    spatial_vectors["right"][i], camera_locations["right"]
                    )
                )
    visualizer.setLandmarks(landmark_spatial_locations)
```

Au niveau de la modélisation avec OpenGL, l'objectif de ce dépôt étant d'étudier le fonctionnement du OAK-D et la bibliothèque DepthAI, je ne vais pas en parler.