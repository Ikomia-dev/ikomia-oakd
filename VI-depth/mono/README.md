# Profondeur - Mono Neural Inference

En Mono Neural Inference, la profondeur est déduite de la carte des disparités. Elle s'obtient en croisant les flux vidéos des caméras, chaque pixel se voit donc attribuer une valeur, celle-ci correspond à la distance qui le sépare de l'OAK-D, elle s'exprime en mètre.

Le programme "depth_map.py" permet d'afficher la disparité avec un code couleur, l'estimation est assez bonne, mais il y a tout de même souvent des pixels isolés, avec une profondeur incohérente.

Pour remédier à ce problème, lorsqu'on cherche à estimer la distance à laquelle se trouve un objet, on fait appel au calculateur de position spatiale. Comme vue dans le point [III](https://github.com/Ikomia-dev/ikomia-oakd/tree/main/III-mono_neural_inference), il suffit de transmettre une zone rectangulaire au calculateur, pour ensuite récupérer la distance moyenne à laquelle elle se situe.

Lors de l'estimation de la distance, le calculateur fait une moyenne des valeurs des pixels de la carte des profondeurs, sur la zone transmise. Cependant, l'objet dont on cherche à estimer sa distance à l'OAK-D a peu de chance de remplir l'intégralité de la zone rectangulaire, l'arrière-plan risque donc d'être pris en compte dans l'estimation.

Heureusement, il existe une solution, il est possible de paramétrer une profondeur minimale et maximale, les pixels avec une valeur qui n'est pas comprise entre les deux ne seront alors pas pris en compte dans la moyenne. Ceci se paramètre en modifiant la configuration de l'objet SpatialLocationCalculator, au même moment qu'au paramétrage de la zone d'intérêt à localiser.

```py
# Chaque pixel à moins de 500 mm ou à plus de 2000 mm de distance est ignoré
slc_conf_data.depthThresholds.lowerThreshold = 500
slc_conf_data.depthThresholds.upperThreshold = 2000
```

Ainsi, en "monovision", il est possible d'estimer efficacement la profondeur avec l'OAK-D, cependant, ça ne remet pas en cause l'utilité de faire de la stéréovision pure. En effet, il faut choisir de faire de l'inférence neuronale stéréo ou mono en fonction de la situation, les deux façons de faire ne rentres pas en concurrence, elles se complètent. 

Pour illustrer ce qu'un mauvais choix peut engendrer, il faut consulter le programme "pose_estimation_mono.py", il reprend l'[exemple](https://github.com/Ikomia-dev/ikomia-oakd/tree/main/_examples) d'estimation de position, mais en faisant de la Mono Neural Inference.