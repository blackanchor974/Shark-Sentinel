# Entraînement

Exécuter les commandes depuis la racine du dépôt. Les données sont privées et ignorées par Git.

```powershell
python train_shark_sentinel.py --video "videos/ma_video.mp4" --domain aerial --extract-every 30
```

Les images vont dans `datasets/shark_sentinel/extracted_frames/aerial/`. Annoter les objets des sept classes définies dans `examples/classes.txt`, en conservant leurs identifiants. Enregistrer les annotations YOLO dans `datasets/shark_sentinel/labels_raw/`, avec le même nom de base que l'image. Utiliser des noms de vidéos uniques pour éviter les collisions.

LabelImg est facultatif : `python -m pip install labelImg`. Le lancer avec :

```powershell
labelImg datasets/shark_sentinel/extracted_frames/aerial examples/classes.txt datasets/shark_sentinel/labels_raw
```

Vérifier que le `classes.txt` du dossier de sauvegarde conserve le même ordre. Un fichier label absent exclut l'image de la préparation. Un fichier vide désigne une image sans aucun objet des classes ciblées ; ne pas employer un label vide pour une image simplement ambiguë.

```powershell
python train_shark_sentinel.py --domain aerial --prepare-only
python train_shark_sentinel.py --domain aerial --train --epochs 80 --imgsz 640
```

`--train` refait la préparation avant l'entraînement. Les sous-dossiers images/train, images/val, labels/train et labels/val du dataset préparé sont remplacés. Conserver les originaux dans extracted_frames et labels_raw.

Le split actuel est aléatoire par image, 80/20, sans graine fixée. Des frames voisines peuvent se retrouver des deux côtés : les métriques peuvent surestimer la généralisation. Réserver des vidéos entières indépendantes pour tester le modèle et prévoir un split par vidéo/session pour une évaluation rigoureuse.

Les poids sont enregistrés sous `runs/shark_sentinel_aerial/` dans le dossier indiqué par Ultralytics. Les entraînements successifs peuvent créer des dossiers numérotés. Fournir explicitement le bon best.pt à la détection.

Pour Colab, compresser le contenu de `datasets/shark_sentinel_aerial/` (images, labels, data.yaml) dans shark_sentinel_aerial.zip et suivre le notebook. Cette archive doit rester hors du dépôt.
