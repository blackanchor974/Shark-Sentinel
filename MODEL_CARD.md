# Modèle aérien Shark Sentinel

- Architecture : YOLO11n, Ultralytics.
- Fichier : models/shark-sentinel-aerial.pt.
- SHA-256 : B445955AFA059B2A81ED46EA51D42F8CC7A96D963C1F286F9E07EA5549D4B849
- Origine : poids locaux runs/shark_sentinel_aerial/train/weights/best.pt, issus de l'entraînement Colab selon l'historique du projet.
- Classes : shark, dolphin, whale, turtle, surfer, swimmer, rock.
- Domaine : vues aériennes ; aucun modèle sous-marin spécialisé publié dans ce dépôt.
- Données : images annotées privées, notamment issues de vidéos ; aucune image ni annotation incluse.

## Évaluation historique

La validation précédemment rapportée portait sur 233 images : mAP50 globale 0,902, mAP50-95 globale 0,582. Pour shark : précision 0,949 et rappel 0,985. Ces résultats n'ont pas été recalculés pour la publication. Le split par image peut partager les mêmes vidéos entre entraînement et validation ; ces chiffres ne constituent pas une mesure indépendante de fiabilité terrain.

## Limites

Confusions possibles avec rochers, dauphins, silhouettes et reflets. Sensibilité aux conditions de mer, à l'altitude, à la lumière et à la taille des objets. Le score de confiance n'est pas une probabilité de sécurité. La confirmation temporelle ne supprime pas les erreurs persistantes. Valider sur des vidéos indépendantes et mesurer les alertes erronées par heure, les événements manqués et le délai d'alerte avant tout usage opérationnel.

## Licence

AGPL-3.0 ; voir LICENSE et la politique Ultralytics : https://www.ultralytics.com/license.
