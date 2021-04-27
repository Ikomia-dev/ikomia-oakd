# Mode [stereo neural inference](https://docs.luxonis.com/en/latest/pages/faq/#id2)
L'idée est de faire tourner un modèle en prennant en entrée les flux vidéos des caméras lattérales, pour calculer précisément la profondeur.
<br><br>


## stream_left_cam
Programme permettant de redimensionner et de diffuser le flux vidéo de la caméra gauche.
<br><br>


## coronamask_lr_cams
Programme permettant de faire tourner le modèle [coronamask](https://github.com/luxonis/depthai-experiments/tree/master/gen2-coronamask) sur les caméras lattérales.

L'objectif maintenant est de déterminer la profondeur à l'aide de ces deux flux vidéos.