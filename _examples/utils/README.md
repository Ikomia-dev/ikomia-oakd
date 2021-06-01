## Fichiers regroupant des fonctionnalités communes aux programmes.
<br>

### OakRunner

classe générique qui peut servir de base à tous nos programmes.
<br><br>


### compute

Regroupe des fonctions pour traiter les données.
- Configurer l'entrée d'un [SpatialLocationCalculator](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.SpatialLocationCalculator)
- Convertir un tableau multidimensionnel en tableau unidimensionnel
- Récupérer des coordonnées vectorielles 3D à partir d'un point 2D
- Calculer le point d'intersection de 2 vecteurs 3D
<br><br>


### draw

Regroupe des fonctions pour écrire/dessiner sur une image.
- Points d'intérêts avec label, confiance et coordonnées spatiales
- Nombre d'images par seconde
<br><br>


### visualize

Regroupe des classes pour visualiser des données en 3D avec [OpenGL](https://www.opengl.org//)
- LandmarksCubeVisualizer : Points d'intérêts centrés avec tracé de la zone d'intérêt (pour observer la forme).
- LandmarksDepthVisualizer : Points d'intérêts placés dans le plan par les intersections des vecteurs entre caméras et points (pour observer la profondeur).
<br><br>


### pose

Fichier de l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose) de Luxonis, il regroupe des fonctions qui permettent de traiter la sortie du modèle [human pose estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_human_pose_estimation_0001.html).