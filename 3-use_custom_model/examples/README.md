# Faire tourner le modèle dans le OAK-D

Des exemples d'utilisation sont disponibles dans les repertoires adjacents, ici, je vais expliquer le fonctionnement général.

Pour utiliser un modèle de notre choix, il faut passer par l'objet [NeuralNetwork](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NeuralNetwork) de la bibliothèque DepthAI, il faut simplement lui passer le modèle et gérer sa sortie de manière appropriée.

```py
nn = pipeline.createNeuralNetwork()
nn.setBlobPath("chemin/vers/le.blob")
```
