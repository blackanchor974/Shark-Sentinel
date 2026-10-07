#!/usr/bin/env python3
"""
Shark Sentinel - Training script

1. Prend une vidéo MP4 en entrée.
2. Extrait des images.
3. Prépare un dataset YOLO mono-domaine (aerial OU underwater).
4. Entraîne un modèle YOLO spécialisé pour ce domaine si les labels existent.

La vidéo seule ne suffit pas : les images extraites doivent être annotées au format YOLO.

Deux modèles séparés sont entraînés (un par domaine) plutôt qu'un modèle unique,
pour correspondre à deux dispositifs distincts (drone aérien / caméra ou drone
sous-marin) qui ne verront chacun qu'un seul type de vue en usage réel :

- Images brutes et annotations : partagées, dans datasets/shark_sentinel/
  (extracted_frames/aerial|underwater/, labels_raw/) — rien à ré-annoter en double.
- Dataset préparé (images/labels train/val + data.yaml) : séparé par domaine,
  dans datasets/shark_sentinel_aerial/ et datasets/shark_sentinel_underwater/.
- Modèle entraîné : séparé par domaine, dans runs/shark_sentinel_aerial/ et
  runs/shark_sentinel_underwater/.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

CLASSES = {
    0: "shark",
    1: "dolphin",
    2: "whale",
    3: "turtle",
    4: "surfer",
    5: "swimmer",
    6: "rock",
}
DOMAINS = ("aerial", "underwater")
RAW_DIR = Path("datasets/shark_sentinel")


def extract_frames(video_path: Path, output_dir: Path, extract_every: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
    frame_index = 0
    saved_count = 0
    stem = video_path.stem
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % extract_every == 0:
            out = output_dir / f"{stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved_count += 1
        frame_index += 1
    cap.release()
    return saved_count


def prepare_domain_dataset(
    domain: str, raw_dir: Path, dataset_dir: Path, train_ratio: float = 0.8
) -> None:
    """Construit un dataset YOLO mono-domaine à partir des images brutes partagées.

    Ne regarde que extracted_frames/<domain>/ : le dataset produit ne contient
    que ce seul type de vue, pour entraîner un modèle spécialisé dessus.
    """
    extracted_dir = raw_dir / "extracted_frames" / domain
    labels_dir = raw_dir / "labels_raw"

    image_files = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in extracted_dir.glob(ext))
    if not image_files:
        raise RuntimeError(f"Aucune image extraite trouvée dans : {extracted_dir}")

    labeled_images = [img for img in image_files if (labels_dir / f"{img.stem}.txt").exists()]
    if not labeled_images:
        raise RuntimeError(
            "Aucun label trouvé pour ce domaine. Annote d'abord les images extraites au format YOLO, "
            f"puis place les .txt dans : {labels_dir}"
        )

    # On repart d'un dossier vide à chaque appel : sans ça, les anciennes
    # répartitions train/val (issues d'appels précédents avec un shuffle
    # différent) s'accumulent et une même image peut finir des deux côtés
    # à la fois (fuite de données entre train et val).
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        sub_dir = dataset_dir / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

    def copy_pair(img: Path, split_name: str):
        lbl = labels_dir / f"{img.stem}.txt"
        shutil.copy2(img, dataset_dir / f"images/{split_name}" / img.name)
        shutil.copy2(lbl, dataset_dir / f"labels/{split_name}" / lbl.name)

    random.shuffle(labeled_images)
    split = int(len(labeled_images) * train_ratio)
    train_images = labeled_images[:split]
    val_images = labeled_images[split:] or labeled_images[-1:]

    for img in train_images:
        copy_pair(img, "train")
    for img in val_images:
        copy_pair(img, "val")

    data = {"path": str(dataset_dir.resolve()), "train": "images/train", "val": "images/val", "names": CLASSES}
    with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Dataset YOLO ({domain}) créé : {dataset_dir}")
    print(f"Images train : {len(train_images)}")
    print(f"Images val   : {len(val_images)}")


def train_model(dataset_dir: Path, base_model: str, epochs: int, imgsz: int, batch: int, project_dir: Path) -> None:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"Fichier data.yaml introuvable : {data_yaml}")
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, project=str(project_dir), name="train",
        patience=20, save=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Shark Sentinel - Train a domain-specific (aerial or underwater) YOLO model"
    )
    parser.add_argument("--video", help="Chemin vers la vidéo MP4 source (optionnel : à omettre si les images sont déjà extraites et annotées)")
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        help="Domaine visé : aerial (drone) ou underwater (plongée/drone sous-marin). "
        "Requis avec --video, --prepare-only ou --train : chaque dataset et chaque modèle sont mono-domaine.",
    )
    parser.add_argument("--extract-every", type=int, default=30, help="Extraire 1 frame toutes les N frames")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dossier du dataset YOLO préparé (images/labels train+val, data.yaml). "
        "Défaut : datasets/shark_sentinel_<domain>/",
    )
    parser.add_argument("--base-model", default="yolo11n.pt", help="Modèle YOLO de départ")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement après préparation du dataset")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prépare le dataset du domaine (trie les images annotées vers images/train, images/val, etc.) "
        "sans lancer l'entraînement.",
    )
    args = parser.parse_args()

    extracted_root = RAW_DIR / "extracted_frames"
    labels_dir = RAW_DIR / "labels_raw"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        if not args.domain:
            parser.error("--domain est requis avec --video (aerial ou underwater)")
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {video_path}")
        print("[1/3] Extraction des frames...")
        domain_dir = extracted_root / args.domain
        saved = extract_frames(video_path, domain_dir, args.extract_every)
        print(f"{saved} images extraites dans : {domain_dir}")
    else:
        print("[1/3] Pas de --video fourni, extraction ignorée (utilisation des images déjà présentes).")

    if not (args.prepare_only or args.train):
        print("\nAnnote les images, puis relance avec --domain <aerial|underwater> --train "
              "(ou --prepare-only pour juste trier).")
        print(f"Images : {extracted_root}/<domain>/")
        print(f"Labels : {labels_dir}")
        return

    if not args.domain:
        parser.error("--domain est requis avec --prepare-only ou --train (aerial ou underwater)")

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(f"datasets/shark_sentinel_{args.domain}")

    if args.prepare_only:
        print(f"Préparation du dataset YOLO ({args.domain})...")
        prepare_domain_dataset(args.domain, RAW_DIR, dataset_dir)
        print(f"\nDataset trié. Relance avec --domain {args.domain} --train quand tu veux entraîner dessus.")
        return

    print(f"[2/3] Préparation du dataset YOLO ({args.domain})...")
    prepare_domain_dataset(args.domain, RAW_DIR, dataset_dir)
    print(f"[3/3] Entraînement du modèle ({args.domain})...")
    project_dir = Path(f"runs/shark_sentinel_{args.domain}").resolve()
    train_model(dataset_dir, args.base_model, args.epochs, args.imgsz, args.batch, project_dir)
    print(f"\nEntraînement terminé : runs/shark_sentinel_{args.domain}/train/weights/best.pt")


if __name__ == "__main__":
    main()
