# Profondeur - Inférence Neuronale Stéréo

En faisant de l'inférence neuronale stéréo, il est possible de calculer précisément la distance qui sépare l'OAK-D de points précis. En monovision, on parle d'estimation de profondeur, ici, on détermine la profondeur.

En stéréovision, pour que croiser les flux vidéos des caméras latérales ait un sens, il faut qu'elles repèrent les mêmes points. Avec la carte de disparité, un algorithme se charge d'estimer quel pixel de la caméra gauche correspond à quel pixel de la caméra droite, malgré qu'il soit assez performant, il reste approximatif. En faisant de l'inférence neuronale sur chaque caméra, on peut faire en sorte que chacune cible les mêmes points, on peut alors calculer précisément la distance entre l'OAK-D et ces points.

Le problème, c'est qu'inférer sur chaque caméra est très coûteux, de plus, il n'est pas envisageable de détecter trop de points, sous peine de trop impacter les performances. Ce mode de fonctionnement a donc vocation à repérer quelques points pour estimer précisément la profondeur. De plus la bibliothèque DepthAI n'aide pas dans la réalisation des calculs, il faut les programmer nous mêmes.

Ces calculs sont assez complexes, dans le point [IV](https://github.com/Ikomia-dev/ikomia-oakd/tree/main/IV-stereo_neural_inference), je me suis inspiré de ceux présent dans l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation) officiel, j'obtiens alors deux fonctions pour déterminer les coordonnées 3D d'un point.

```py
# Récupère la direction du vecteur allant de la caméra au point
def get_landmark_3d(landmark, focal_length=842, size=640):
    landmark_norm = 0.5 - np.array(landmark)
    landmark_image_coord = landmark_norm * size

    landmark_spherical_coord = [math.atan2(landmark_image_coord[0], focal_length),
                                -math.atan2(landmark_image_coord[1], focal_length) + math.pi / 2]

    landmarks_3D = [
        math.sin(landmark_spherical_coord[1]) * math.cos(landmark_spherical_coord[0]),
        math.sin(landmark_spherical_coord[1]) * math.sin(landmark_spherical_coord[0]),
        math.cos(landmark_spherical_coord[1])
    ]

    return landmarks_3D


# Détermine l'intersection entre les vecteurs directionnels.
def get_vector_intersection(left_vector, left_camera_position, right_vector, right_camera_position):
    if(len(left_vector)<2 or len(right_vector)<2):
        return []
    n = np.cross(left_vector, right_vector)
    n1 = np.cross(left_vector, n)
    n2 = np.cross(right_vector, n)

    top = np.dot(np.subtract(right_camera_position, left_camera_position), n2)
    bottom = np.dot(left_vector, n2)
    divided = top / bottom
    mult = divided * left_vector
    c1 = left_camera_position + mult

    top = np.dot(np.subtract(left_camera_position, right_camera_position), n1)
    bottom = np.dot(right_vector, n1)
    divided = top / bottom
    mult = divided * right_vector
    c2 = right_camera_position + mult

    center = (c1 + c2) / 2
    return center
```

Ceci semble fonctionnel, j'ai testé en réalisant un programme "pose_estimation_stereo.py" qui détecte la forme d'un corps humain, puis, la modélise en 3D. Cependant, je ne comprends pas les calculs, je me suis donc penché sur une autre façon de faire en m'aidant de cet [article](https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/).

L'idée est de faire l'opération inverse d'une capture d'image, au lieu de projeter la scène 3D en 2D (dans le plan image), on va reprendre les calculs pour passer de la projection 2D au coordonnées 3D. Pour en savoir plus sur ce processus, voici un [article](https://learnopencv.com/geometry-of-image-formation/) assez complet.

Pour ce faire, il faut récupérer les caractéristiques intrinsèques et extrinsèques des caméras et heureusement pour nous, ces informations sont stockées dans l'OAK-D, il suffit d'instancier la classe [CalibrationHandler](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.CalibrationHandler), pour illustrer mes propos, voici quelques extraits du programme "calibration_info.py".

```py
# Création d'un objet CalibrationHandler
calibration = device.readCalibration()

# Récupération des paramètres intrinsèques
left_intrinsics = np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
right_intrinsics = np.array(calibration.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))

# Récupération des paramètres extrinsèques
extrinsics = np.array(calibration.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
```

À partir des matrices intrinsèques et extrinsèques, il est possible de calculer les matrices de projection des caméras, il s'agit d'un simple produit matriciel.
```py
projection_matrix = intrinsic_matrix.dot(extrinsic_matrix)
```

Enfin, à partir de cette matrice de projection, il est théoriquement possible d'en déduire un vecteur pointant vers le point dans l'espace 3D, j'ai donc écris une fonction qui est censé remplir le rôle de get_landmark_3d.
```py
def getSpatialVector(x, y, projection_matrix):
    point = np.array([x,y,1]).reshape(3, 1).astype(np.float32)
    vector = np.linalg.pinv(projection_matrix).dot(point)
    return [vector[0][0], vector[1][0], vector[2][0]]
```

Cependant, en reproduisant le programme d'estimation de position 3D, "pose_estimation_stereo_alternative.py", je me suis aperçu que mes calculs sont totalement erronés. Je pense néanmoins qu'il ne faut pas écarter cette piste, il me semble plus juste de calculer la position 3D à partir des matrices de projection des caméras.

Pour en revenir à l'inférence neuronale stéréo, comme précisé dans la partie traitant de la monovision, il faut choisir de faire de l'inférence neuronale stéréo ou mono en fonction de la situation, les deux façons de faire ne rentres pas en concurrence, elles se complètent. Ainsi, pour visualiser la forme d'un objet, il vaut mieux faire de l'inférence neuronale stéréo, tandis que pour simplement estimer sa distance par rapport au dispositif, il vaut mieux inférer uniquement sur la caméra centrale et se fier à la carte de disparité.