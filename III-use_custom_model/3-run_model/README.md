# Faire tourner le modèle dans le OAK-D

Pour utiliser un modèle de notre choix, il faut se servir des objets proposés par la [bibliothèque](https://docs.luxonis.com/projects/api/en/latest/references/python/) DepthAI, je vais expliquer pourquoi et comment se servir d'eux.

Pour se faire, j'ai créé plusieurs programmes, ils font tous la même chose, détecter les visages et indiquer si un masque est porté ([modèle](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask)).

Ainsi, j'ai créé 3 dossiers, un par approche utilisée pour le traitement, il en existe d'autres, mais, pour la détection, ce sont les plus utiles (selon moi).
<br><br>


## Réseau de neurones quelconque

Pour utiliser un modèle de notre choix, la façon la plus générique de le faire en en passant par l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork), il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

Gérer la sortie s'avère souvent plus complexe qu'avec des objets plus spécifiques, cependant, avec une instance NeuralNetwork, il est certain que le modèle pourra être utilisé (en supposant qu'il soit au bon format pour la Myriade X).

Des exemples d'utilisations sont disponibles dans le dossier "as_neuralNetwork".
<br><br>


## Réseau de neurones de détection spécifique

Certains objets sont très spécifiques, il en existe beaucoup, le fonctionnement est souvent le même, je ne vais donc en présenter qu'un, celui de l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork).

Comme son nom l'indique, l'objet [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork) est plus spécifique que NeuralNetwork et DetectionNetwork, il n'est utilisable qu'avec des modèles basés sur [MobileNet-SSD](https://docs.openvinotoolkit.org/latest/omz_models_model_mobilenet_ssd.html). Son intéret est qu'il simplifie beaucoup l'utilisation d'un modèle compatible par rapport aux objets plus génériques.

Des exemples d'utilisations sont disponibles dans le dossier "as_specificDetectionNetwork".