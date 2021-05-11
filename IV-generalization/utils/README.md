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