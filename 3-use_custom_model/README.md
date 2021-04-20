# Utiliser un modèle personnalisé dans le OAK-D

Il est necessaire d'utiliser [OpenVino](https://docs.openvinotoolkit.org/latest/index.html) pour compiler un modèle de sorte à ce qu'il soit compatible avec le OAK-D, pour l'installation, les tutoriels [Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_get_started_get_started_windows.html) et [Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_get_started_get_started_linux.html) sont très clairs.

Néanmoins, pour la compatibilité avec le OAK-D, il faut utiliser une version d'OpenVino entre [2020.1](https://docs.openvinotoolkit.org/2020.1/index.html) et [2021.2](https://docs.openvinotoolkit.org/2021.2/index.html). En l'occurence, je vais utiliser la version [2021.1](https://docs.openvinotoolkit.org/2021.2/index.html) car elle permet de contourner les contraintes d'hardware (processeur intel), j'y reviendrai plus tard.
<br><br>


## 1. Convertir le modèle en [représentation intermédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) d'OpenVino

Avant de pouvoir compiler le fichier en un format compréhensible par la [Myriade X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html), il faut passer par une représentation intermédiaire, grossièrement, un fichier xml et un fichier bin. Pour se faire, voici un petit tutoriel inspiré de l'[officiel](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) et de celui de [Luxonis](https://docs.luxonis.com/projects/api/en/latest/tutorials/local_convert_openvino/).

Par défaut, OpenVino offre des scripts python permettant de convertir 5 types de modèles différents en une RI (représentation intermédiaire) :
1. [caffe](https://caffe.berkeleyvision.org/)
2. [kaldi](https://kaldi-asr.org/)
3. [mxnet](https://mxnet.apache.org/versions/1.8.0/) (pour les modèles créés à partir d'une version de mxnet entre 1.0.0 et 1.5.1)
4. [onnx](https://onnx.ai/) (pour les modèles créés avec au moins la version 1.1.2)
5. [tensorflow](https://www.tensorflow.org/?hl=fr) (pour les modèles créés à partir d'une version de tensorflow supérieure ou égale à 1.2.0 et inférieure à 2.0.0)

Pour convertir un modèle, il suffit alors d'appeler le script correspondant, prennons un modèle tensorflow "modele.pb" ([exemple officiel](https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYolov3_Object_Detector_Training_on_Custom_Data.ipynb)):

Supposons que le chemin d'accès vers les scripts soit "chemin" :
- Sous Windows : 'C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/model_optimizer'
- Sous Linux : '/opt/intel/openvino/deployment_tools/model_optimizer'

Pour convertir le modèle tensorflow en RI il suffit d'appeler le script "mo_ts.py"
```
python chemin/mo_ts.py 
    --input_model modele.pb # modèle au format tensorflow
    --tensorflow_use_custom_operations_config modele.json  # json décrivant la sortie du modèle
    --batch 1 
    --data_type FP16 # type de la sortie
    --output_dir ./
```

Les fichiers .xml et .bin sont alors générés.
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

Cela se fait grâce à l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) de la bibliothèque DepthAI, il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

```py
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")
```

Un exemple est disponibles dans le dossier "run_samples".