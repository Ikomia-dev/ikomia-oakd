# Mode [stereo neural inference](https://docs.luxonis.com/en/latest/pages/faq/#id2)

L'idée est de faire tourner un modèle en prennant en entrée les flux vidéos des caméras lattérales, pour calculer précisément la profondeur en croisant leurs flux.

Pour se faire, il faut s'assurer que chacune des caméras repère les mêmes points, sinon, croiser leurs flux n'aurait pas de sens. C'est pourquoi, les programmes qui vont suivrent s'appuyerons sur l'identification de points d'intérêts du visage. En effet, admétons qu'il y est une seule personne dans le champ visuel, si sur chacune des caméras je repère un nez, alors je peux croiser les flux et donc récupérer les coordonnées spatiales.

Afin de m'aider à réaliser ceci, je me suis aidé d'un [exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation) de Luxonis (presque l'unique exemple utilisant l'inférence neuronale stéréo).
<br><br>


## face_detector
Programme permettant de détecter un visage (celui avec le plus de confience), à partir des flux vidéos des caméras lattérales et à l'aide de ce [modèle](https://github.com/luxonis/depthai-experiments/blob/master/gen2-triangulation/models/face-detection-retail-0004_2021.3_6shaves.blob).
<br><br>


## face_landmarks_detector
Programme similaire, mais qui, à l'intérieur de la zone du visage, va identifier 5 points d'intérêts (yeux, nez, bouche) à l'aide de ce [modèle](https://github.com/luxonis/depthai-experiments/blob/master/gen2-triangulation/models/landmarks-regression-retail-0009_2021.3_6shaves.blob).
<br><br>


## face_visualizer
Programme similaire, mais qui, à partir des points d'intérêts detectés, calcule les coordonnées spatiales et visualise les points d'intérêts en 3D (grâce à [pygame](https://www.pygame.org/news) et [OpenGL](https://www.opengl.org//))