# Scripts de compilation


## Générer la RI ([représentation intermmédiaire](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html))
<br>

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
<br><br>


## Compiler la RI en un format compréhensible par la [Myriade X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html) (.blob)

```
python compile_ir model.xml model.bin compiled_model.blob
```
Génère le .blob (dans le repertoire actuel si seul le nom du fichier est spécifié, sinon, prend les chemins indiqués).