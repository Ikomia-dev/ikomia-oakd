# Utilisation des ressources lors de l'exécution

Voulant m'assurer qu'à l'exécution, les ressources du OAK-D sont utilisées (et pas seulement celle de ma machine), j'ai fait 2 programmes, qui, après exécution, affiche des graphiques montrant l'évolution du pourcentage d'utilisation des ressources (cpu / ram) du OAK-D et de la machine hôte.

Le premier se contente d'afficher le flux vidéo de la caméra, le second utiliser l'algorithme tiny [YOLO](https://appsilon.com/object-detection-yolo-algorithm/) de l'exemple [22_2](https://github.com/luxonis/depthai-python/blob/main/examples/22_2_tiny_yolo_v4_device_side_decoding.py).

Dans le dossier graphs, j'ai déposé quelques captures d'écrans des graphiques obtenus suite à l'utilisation de mes programmes, j'ai pu remarqué que les ressources du OAK-D sont bien utilisées.

Enfin, ces programmes me seront utiles par la suite pour établir un benchmark de l'appareil, ainsi, même si pour l'instant, l'utilité de ceux-ci est relative, à l'avenir, elle ne fait aucun doute.