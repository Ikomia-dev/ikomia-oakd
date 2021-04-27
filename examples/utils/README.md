## Fichiers regroupant des fonctionnalités communes aux programmes.
<br>

### OakSingleModelRunner

classe générique qui peut servir de base à tous nos programmes.
<br><br>

### draw

Regroupe des fonctions pour écrire/dessiner sur une image.
- Points d'intérêts avec label, confience et coordonnées spatiales
- Nombre d'images par seconde
<br><br>


### compute

Regroupe des fonctions pour traiter les données.
- Configurer l'entrée d'un [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator)
<br><br>


### pose

Fichier de l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose) de Luxonis, il regroupe des fonctions qui permettent de traiter la sortie du modèle [human pose estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_human_pose_estimation_0001.html).