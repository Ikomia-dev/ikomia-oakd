# Faire tourner le modèle dans le OAK-D

Pour utiliser un modèle de notre choix, il faut se servir des objets proposés par la [bibliothèque](https://docs.luxonis.com/projects/api/en/latest/references/python/) DepthAI, je vais expliquer pourquoi et comment se servir d'eux.

Pour se faire, j'ai créé plusieurs programmes assez similaire, dans 3 dossiers, un par approche possible pour le traitement, elles reposent principalement sur ces 3 objets :
- [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) : approche pour un modèle quelconque.
- [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork) : approche pour un modèle MobileNet.
- [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork) : approche pour un modèle YOLO.
<br><br>


## Réseau de neurones quelconque

Pour utiliser un modèle de notre choix, la façon la plus générique de le faire en en passant par l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork), il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

Gérer la sortie s'avère souvent plus complexe qu'avec des objets plus spécifiques, cependant, avec une instance NeuralNetwork, il est certain que le modèle pourra être utilisé (en supposant qu'il soit au bon format pour la Myriade X).

Des explications plus détaillées sont disponibles dans le dossier "as_neuralNetwork".
<br><br>


## Réseau de neurones [MobileNet](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html)

Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [MobileNet](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html), [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.

Pour pouvoir obtenir les coordonnées spatiales, il faut privilégier l'objet [MobileNetSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetSpatialDetectionNetwork), il fait implicitement le travail d'un calculateur de localisation spatiale.

Des explications plus détaillées sont disponibles dans le dossier "as_mobilenetDetectionNetwork".
<br><br>


## Réseau de neurones [YOLO](https://appsilon.com/object-detection-yolo-algorithm/)

Il existe un objet plus spécifique que NeuralNetwork pour utiliser un modèle de détection [YOLO](https://appsilon.com/object-detection-yolo-algorithm/), [YoloDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloDetectionNetwork), son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport à l'objet générique.

Pour pouvoir obtenir les coordonnées spatiales, il faut privilégier l'objet [YoloSpatialDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.YoloSpatialDetectionNetwork), il fait implicitement le travail d'un calculateur de localisation spatiale.

Des explications plus détaillées sont disponibles dans le dossier "as_yoloDetectionNetwork".