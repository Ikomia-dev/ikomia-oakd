# Profondeur - Inférence Neuronale Stéréo

En faisant de l'inférence neuronale stéréo, il est possible de calculer précisément la distance qui sépare l'OAK-D de points précis. En monovision, on parle d'estimation de profondeur, ici, on détermine la profondeur.

En stéréovision, pour que croiser les flux vidéos des caméras latérales ait un sens, il faut qu'elles repèrent les mêmes points. Avec la carte de disparité, un algorithme se charge d'estimer quel pixel de la caméra gauche correspond à quel pixel de la caméra droite, malgré qu'il soit assez performant, il reste approximatif. En faisant de l'inférence neuronale sur chaque caméra, on peut faire en sorte que chacune cible les mêmes points, on peut alors calculer précisément la distance entre l'OAK-D et ces points.

Le problème, c'est qu'inférer sur chaque caméra est très coûteux, de plus, il n'est pas envisageable de détecter trop de points, sous peine de trop impacter les performances. Ce mode de fonctionnement a donc vocation à repérer quelques points pour estimer précisément la profondeur. De plus la bibliothèque DepthAI n'aide pas dans la réalisation des calculs, il faut les programmer nous mêmes.

Afin de comprendre les calculs, voici un [article](https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/) qui explique la théorie de la stéréovision. Sinon, en m'inspirant des calculs présent dans l'[exemple](https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation), j'ai écris deux fonctions pour déterminer les coordonnées 3D d'un point.

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


Pour montrer un cas d'utilisation, j'ai réalisé un programme "pose_estimation.py" qui détecte la forme d'un corps humain, puis, la modélise en 3D.

Enfin, comme précisé dans la partie traitant de la monovision, il faut choisir de faire de l'inférence neuronale stéréo ou mono en fonction de la situation, les deux façons de faire ne rentres pas en concurrence, elles se complètent. Ainsi, pour visualiser la forme d'un objet, il vaut mieux faire de l'inférence neuronale stéréo, tandis que pour simplement estimer sa distance par rapport au dispositif, il vaut mieux inférer uniquement sur la caméra centrale et se fier à la carte de disparité.