# Découverte OAK-D et DepthAI

L'[OAK-D](https://docs.luxonis.com/en/latest/pages/products/bw1098obc/) est un produit [Luxonis](https://docs.luxonis.com/en/latest/) combinant carte de programmation et caméras, il vise à mettre à profit le deep learning pour analyser l'environnement en temps réel, il permet de le visualiser en 3D.
<br><br>


## Dispositif

L'OAK-D est un dérivé du [LUX-D](https://drive.google.com/file/d/1g0bQDLNnpVC_1-AGaPmC8BaXtGaNNdTR/view "fiche technique LUX-D"), il sagit simplement de sa version avec un support physique associé (une protection autour de la carte).

L'appareil possède 3 caméras, une centrale filmant jusqu'en 4K avec un taux de rafraichissement de 60Hz et deux lattérales, filmant jusqu'en 720p avec un taux de rafraichissement de 120Hz. L'idée, c'est que la caméra centrale sert à capturer le flux vidéo, tandis que les caméras lattérales servent à capturer la profondeur de la scène (grâce au croisement de leurs flux)

Au niveau du processeur, celui utilisé est l'[Intel® Movidius™ Myriad™ X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html), d'après la [fiche technique](https://drive.google.com/file/d/1z7QiCn6SF3Yx977oH41Kcq68Ay6e9h3_/view "fiche technique modèles de DepthAI"), il s'agit plus précisément du [MA2485](https://ark.intel.com/content/www/us/en/ark/products/125926/intel-movidius-myriad-x-vision-processing-unit-4gb.html) ou du [MA2085](https://ark.intel.com/content/www/us/en/ark/products/204770/intel-movidius-myriad-x-vision-processing-unit-0gb.html).
<br><br>



## Utilisation

L'OAK-D est utilisable sous les principaux OS, entre autre, il suffit d'avoir un environnement de developpement Python, un compilateur C++, CMake et la [bibliothèque DepthAI](https://docs.luxonis.com/projects/api/en/latest/install/) associée, plus de détails sont disponible dans le dossier "depthai_library".
<br><br>


## Ressources

Les ressources du OAK-D sont utilisées pour l'inférence neuronale, ceci peut s'observer dans le dossier "device_information".