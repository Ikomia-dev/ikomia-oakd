# Conversion d'un modèle vers une RI ([représentation intermmédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html))


## Environnement de travail
Au début, j'ai cru qu'il fallait utiliser la version [2020.1](https://docs.openvinotoolkit.org/2020.1/index.html) d'OpenVino, cependant, les sources sur lesquelles je m'étais basé sont désormais obselettes, j'ai perdu du temps, mais ça reste une bonne nouvelle.

De plus, pour convertir un modèle tensorflow en RI, il faut absolument utiliser une version de tensorflow [inférieure](https://www.tensorflow.org/install/pip?hl=fr#older-versions-of-tensorflow) à la [2.0.0](https://www.tensorflow.org/), mais, celle-ci n'est pas disponible sous la dernière version de Python, cela impose donc de revenir à sa version [3.6](https://www.python.org/downloads/release/python-360/).

Avec toutes ces contraintes j'ai décidé de créer un environnement virtuel à l'aide de docker. Cependant, la création des scripts de conversion de modèles étant mis en pause pour le moment, je ne l'ai pas fini, de plus, l'environnement utilise OpenVino [2020.1](https://docs.openvinotoolkit.org/2020.1/index.html) alors que la version [2021.1](https://docs.openvinotoolkit.org/2021.1/index.html) est plus adaptée.

Néanmoins, ayant préféré push le Dockerfile, voici un petit tutoriel d'utilisation :
```sh
# Créer l'image docker
docker build -t convert_ir_venv . # Depuis le répertoire contenant le Dockerfile

# Lance le conteneur et créer un lien avec l'hôte
docker run -dit -P convert_ir_venv -v ./venv:/venv 

# Liste les conteneurs en cours, permet de récupérer l'id ID
docker ps

# Rentre dans le bash du conteneur
docker exec -it ID /bin/bash 
```
<br>


## Conversion

### [caffe](https://caffe.berkeleyvision.org/)
```
script en cours d'écriture
```
<br>

### [kaldi](https://kaldi-asr.org/)
```
script en cours d'écriture
```

### [mxnet](https://mxnet.apache.org/versions/1.8.0/)
```
script en cours d'écriture
```

### [onnx](https://onnx.ai/)
```
script en cours d'écriture
```

### [tensorflow](https://www.tensorflow.org/?hl=fr)
```
script en cours d'écriture
```