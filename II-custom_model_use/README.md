# Utiliser un modèle personnalisé dans le OAK-D

Il est necessaire d'utiliser [OpenVino](https://docs.openvinotoolkit.org/latest/index.html) pour compiler un modèle de sorte à ce qu'il soit compatible avec l'OAK-D, pour l'installation, les tutoriels [Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_get_started_get_started_windows.html) et [Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_get_started_get_started_linux.html) sont très clairs.

Néanmoins, pour la compatibilité avec l'OAK-D, il faut utiliser une version d'OpenVino entre [2020.1](https://docs.openvinotoolkit.org/2020.1/index.html) et [2021.2](https://docs.openvinotoolkit.org/2021.2/index.html). En l'occurence, je vais utiliser la version [2021.1](https://docs.openvinotoolkit.org/2021.2/index.html) car elle permet de contourner les contraintes d'hardware (processeur intel), j'y reviendrai plus tard.
<br><br>


## 1. Convertir le modèle en [représentation intermédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) d'OpenVino

Avant de pouvoir compiler le fichier en un format compréhensible par la [Myriade X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html), il faut passer par une représentation intermédiaire, grossièrement, un fichier xml et un fichier bin.

Seulement 5 formats de modèles peuvent être convertis : [caffe](https://caffe.berkeleyvision.org/), [kaldi](https://kaldi-asr.org/), [mxnet](https://mxnet.apache.org/versions/1.8.0/), [onnx](https://onnx.ai/) et [tensorflow](https://www.tensorflow.org/), pour utiliser un autre format, il faut au préalable le convertir en l'un d'entre eux.
<br><br>



## 2. Compiler la représentation intermédiaire en fichier .blob

Pour cette étape, techniquement, il faut utiliser une machine avec un processeur intel (d'au moins la 6ème génération), cependant, Luxonis propose une [API](http://69.164.214.171:8083/) se chargeant de la compilation, il suffit donc de l'utiliser.

Pour simplifier l'automatisation de la chose, j'ai fais un script qui appel cette API, voici un exemple d'utilisation :

```
python compile_ir model.xml model.bin compiled_model.blob
```
Génère le .blob (dans le repertoire actuel si seul le nom du fichier est spécifié, sinon, prend les chemins indiqués).
<br><br>



## 3. Faire tourner le modèle dans le OAK-D

Cela se fait en instanciant des objets proposés par la bibliothèque DepthAI, le plus générique d'entre eux est [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork), il faut lui passer le modèle et gérer la sortie de manière appropriée.

```py
# Instanciation du réseau de neurones
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")
```

Cependant, il existe beaucoup d'autres objets plus spécifiques, comme [MobileNetDetectionNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.MobileNetDetectionNetwork), l'intérêt est que la sortie de l'instance d'un objet plus spécifique est souvent plus simple à traiter.

Dans le dossier "run_model", l'utilisation de diverses objets est montré via la réalisation d'un même programme de plusieurs façons différentes.