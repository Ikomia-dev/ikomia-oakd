# Conversion d'un modèle vers une RI ([représentation intermmédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html))

Pour se faire, voici un petit tutoriel inspiré de l'[officiel](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) et de celui de [Luxonis](https://docs.luxonis.com/projects/api/en/latest/tutorials/local_convert_openvino/).

Par défaut, OpenVino offre des scripts python permettant de convertir 5 types de modèles différents en une RI (représentation intermédiaire) :
1. [caffe](https://caffe.berkeleyvision.org/)
2. [kaldi](https://kaldi-asr.org/)
3. [mxnet](https://mxnet.apache.org/versions/1.8.0/) (pour les modèles créés à partir d'une version de mxnet entre 1.0.0 et 1.5.1)
4. [onnx](https://onnx.ai/) (pour les modèles créés avec au moins la version 1.1.2)
5. [tensorflow](https://www.tensorflow.org/?hl=fr) (pour les modèles créés à partir d'une version de tensorflow supérieure ou égale à 1.2.0 et inférieure à 2.0.0)

Pour convertir un modèle, il suffit alors d'appeler le script correspondant, prennons un modèle tensorflow "modele.pb" ([exemple luxonis](https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/Easy_TinyYolov3_Object_Detector_Training_on_Custom_Data.ipynb)):

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