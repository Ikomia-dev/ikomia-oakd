# Compiler la RI en un format compréhensible par la [Myriade X](https://www.intel.fr/content/www/fr/fr/products/details/processors/movidius-vpu/movidius-myriad-x.html) (.blob)

```
python compile_ir model.xml model.bin compiled_model.blob
```
Génère le .blob (dans le repertoire actuel si seul le nom du fichier est spécifié, sinon, prend les chemins indiqués).