# Faire tourner un modèle personnalisé dans le OAK-D

Cela se fait grâce à l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) de la bibliothèque DepthAI, il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

```py
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")
```


J'ai créé un programme exemple à partir de ce modèle de réseau de neurones :
- [medmask](https://github.com/luxonis/depthai-ml-training/tree/master/model-zoo/medmask) : Détection de port ou non du masque.