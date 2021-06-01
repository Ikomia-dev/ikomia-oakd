# Profondeur - Mono Neural Inference

Dans le mode de fonctionnement monovision, la profondeur est estimé en moyenant la disparité des pixels de la zone d'intérêt souhaitée. Il s'agit simplement d'une lecture de la carte des profondeurs, pour visualiser sa précision, il suffit de l'afficher.

Le programme "depth_map.py" permet donc d'afficher la disparité avec un code couleur, l'estimation est assez bonne.

Il faut evidemment choisir de faire de l'inférence neuronale stéréo ou mono, en fonction de la situation, les deux façons de faire ne rentres pas en concurrence, elles se complètent. Pour illustrer ce qu'un mauvais choix peut engendrer, il faut consulter le programme "pose_estimation_mono", il reprend l'[exemple](https://github.com/Ikomia-dev/ikomia-oakd/tree/main/_examples) d'estimation de position, mais en faisant de la Mono Neural Inference.