# Utiliser un modèle personnalisé dans le OAK-D

Il est necessaire d'utiliser la version 2.1 d'OpenVino pour compiler un modèle de sorte à ce qu'il soit compatible avec le OAK-D, les tutoriels d'installations [Windows]() et [Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html) sont très clairs.
<br><br>


## 1. Convertir le modèle en [représentation intermédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) d'OpenVino

Avant de pouvoir compiler le fichier en un format compréhensible pour la Myriade X, il faut passer par une représentation intermédiaire, grossièrement, un fichier xml et un fichier bin. Pour se faire, voici un petit tutoriel inspiré de l'[officiel](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) et de celui de [Luxonis](https://docs.luxonis.com/projects/api/en/latest/tutorials/local_convert_openvino/).

Par défaut, OpenVino offre des scripts python permettant de convertir 5 types de modèles différents en une RI (représentation intermédiaire) :
1. caffe
2. kaldi
3. mxnet (réalisé à partir d'une version de mxnet entre 1.0.0 et 1.5.1)
4. onnx (pour les modèles créés avec au moins la version 1.1.2)
5. tensorflow (réalisé à partir d'une version de tensorflow supérieure ou égale à 1.2.0 et inférieure à 2.0.0)

Pour convertir un modèle, il suffit alors d'appeler le script correspondant, par exemple, pour un modèle tensorflow "modele.pb" ([exemple officiel](https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYolov3_Object_Detector_Training_on_Custom_Data.ipynb)):

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

Pour fonctionner avec le OAK-D, la représentation intermédiaire doit avoir été créée à partir d'une version 2020.1 d'OpenVino, le script ne laisse donc pas le choix de la version.

```
python compile_ir modele.xml modele.bin modele_compile.blob
```
Génère le .blob (dans le repertoire actuel si seul le nom du fichier est spécifié).



## 3. Faire tourner le modèle dans le OAK-D

Il suffit d'initialiser l'objet NeuralNetwork de la bibliothèque DepthAI, lui passer le modèle et gérer sa sortie.
```py
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le/blob")
```

Un exemple est disponibles dans le dossier "run_samples".