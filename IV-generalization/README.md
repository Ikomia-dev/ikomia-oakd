# Généralisation de l'utilisation du OAK-D

Il y a régulièrement des répétitions entre les programmes qui font tourner un modèle sur le OAK-D, c'est pourquoi il me semble important de commencer à créer des fichiers qui regroupent les fonctionnalités communes.

L'idée est de se créer un genre de petite bibliothèque, cela va faire gagner beaucoup de temps. Des exemples d'utilisations sont disponibles dans le répertoire adjacent.
<br><br>


## Runner

J'ai créé une classe générique qui peut servir de base à nos programmes, l'instancier permet d'avoir un "runner", il suffit de le configurer via l'appel de quelques méthodes, puis, utiliser la méthode "run" en passant en paramètre la fonction appliquant le traitement souhaité.

![diagramme de classe simplifié](runner_simplified_class_diagram.png "diagramme de classe simplifié")

Cela permet d'écrire un programme en très peu de lignes, voici par exemple, un programme équivalent au [mdn_coronamask](https://github.com/Ikomia-dev/ikomia-oakd/blob/main/III-use_custom_model/3-run_model/as_mobilenetDetectionNetwork/mdn_coronamask.py) du point [III.3](https://github.com/Ikomia-dev/ikomia-oakd/tree/main/III-use_custom_model/3-run_model) (qui comptabilise environ 80 lignes).

```py
from OakSingleModelRunner import OakSingleModelRunner
import cv2

# Fonction appelée à chaque itération de la boucle de traitement
def process(runner):
    w = runner.middle_cam.getPreviewWidth()
    h = runner.middle_cam.getPreviewHeight()
    frame = runner.middle_cam_output_queue.get().getCvFrame()
    detections = runner.nn_output_queue.get().detections
    for det in detections:
        topleft = (int(det.xmin*w), int(det.ymin*h))
        botright = (int(det.xmax*w), int(det.ymax*h))
        cv2.rectangle(frame, topleft, botright, (255,0,0), 2)
        cv2.putText(frame, runner.labels[det.label] + f" {int(det.confidence*100)}%", (topleft[0]+10, topleft[1]+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,0,0))
    cv2.imshow("output", frame)

# Instancier le runner
runner = OakSingleModelRunner() 

# Configurer la caméra centrale
runner.setMiddleCamera(frame_width=300, frame_height=300)
runner.middle_cam.setInterleaved(False)
runner.middle_cam.setFps(20)

# Configurer le réseau de neurones
runner.setMobileNetDetectionModel(path="/chemin/vers/le.blob")
runner.labels = ["background", "no mask", "mask", "no mask"]

# Exécuter la boucle de traitement
runner.run(process=process)
```
<br><br>


## Fonctionnalités communes

Le runner permet de gagner beaucoup de temps, mais il est possible de faire mieux. Pour se faire je me lance dans l'écriture de fichiers regroupant des fonctionnalités communes aux programmes (affichage ROI/label, affichage fps...).

Une fois cela fait, je réécrirai le programme ci-dessus, avec encore moins de lignes.

(à suivre...)