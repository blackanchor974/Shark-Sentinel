#!/usr/bin/env python3
"""Shark Sentinel - Detection script (image, dossier d'images ou vidéo)."""

import argparse
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

SHARK_CLASS_NAMES = {"shark", "bull_shark", "tiger_shark"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def count_shark_detections(result, model: YOLO) -> int:
    count = 0
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, str(class_id)).lower()
        if class_name in SHARK_CLASS_NAMES:
            count += 1
    return count


def frame_has_confident_shark(result, model: YOLO, alert_conf: float) -> bool:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, str(class_id)).lower()
        if class_name in SHARK_CLASS_NAMES and float(box.conf[0]) >= alert_conf:
            return True
    return False


class AlertTracker:
    """Déclenche une alerte seulement si le requin est détecté de façon persistante.

    Une détection isolée sur une seule frame (rocher, reflet, surfeur) ne suffit pas :
    il faut au moins `min_hits` frames positives (confiance >= alert_conf) parmi les
    `window` dernières frames *analysées* (les frames sautées via --frame-skip ne
    comptent pas, faute de résultat frais). Un vrai requin reste visible sur plusieurs
    frames consécutives, un faux positif ponctuel beaucoup plus rarement.
    """

    def __init__(self, window: int, min_hits: int, alert_conf: float):
        if min_hits > window:
            raise ValueError("--alert-min-hits ne peut pas dépasser --alert-window")
        self.window = window
        self.min_hits = min_hits
        self.alert_conf = alert_conf
        self.history = deque(maxlen=window)
        self.active = False

    def update(self, result, model: YOLO) -> str | None:
        """Enregistre le résultat d'une frame analysée. Retourne 'start', 'end' ou None."""
        self.history.append(frame_has_confident_shark(result, model, self.alert_conf))

        was_active = self.active
        self.active = len(self.history) == self.window and sum(self.history) >= self.min_hits

        if self.active and not was_active:
            return "start"
        if was_active and not self.active:
            return "end"
        return None


def draw_alert_banner(frame, active: bool):
    if not active:
        return frame
    height, width = frame.shape[:2]
    text = "ALERTE REQUIN"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 255), thickness=-1)
    x = (width - text_width) // 2
    y = (50 + text_height) // 2
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness=thickness)
    return frame


def detect_on_video(
    video_path: Path,
    model: YOLO,
    output_path: Path,
    confidence: float,
    display: bool,
    imgsz: int | None = None,
    frame_skip: int = 1,
    alert_tracker: "AlertTracker | None" = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    predict_kwargs = {"conf": confidence, "verbose": False}
    if imgsz:
        predict_kwargs["imgsz"] = imgsz

    frame_index = 0
    shark_detections = 0
    alert_events = 0
    last_annotated = None
    print("Détection en cours...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            result = model.predict(source=frame, **predict_kwargs)[0]
            annotated = result.plot()
            shark_detections += count_shark_detections(result, model)
            if alert_tracker is not None:
                event = alert_tracker.update(result, model)
                if event == "start":
                    alert_events += 1
                    timestamp = frame_index / fps
                    print(f"  [ALERTE] requin confirmé à {timestamp:.1f}s (frame {frame_index})")
                elif event == "end":
                    timestamp = frame_index / fps
                    print(f"  [fin d'alerte] à {timestamp:.1f}s (frame {frame_index})")
            last_annotated = annotated
        else:
            # Frame sautée : on réaffiche la dernière détection connue au lieu de relancer le modèle.
            annotated = last_annotated if last_annotated is not None else frame

        if alert_tracker is not None:
            annotated = draw_alert_banner(annotated, alert_tracker.active)

        writer.write(annotated)

        if display:
            cv2.imshow("Shark Sentinel - Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("\nAnalyse terminée.")
    print(f"Vidéo de sortie : {output_path}")
    print(f"Frames analysées : {frame_index}")
    print(f"Détections requin : {shark_detections}")
    if alert_tracker is not None:
        print(f"Alertes déclenchées : {alert_events}")


def detect_on_image(
    image_path: Path, model: YOLO, output_path: Path, confidence: float, display: bool, imgsz: int | None = None
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Impossible d'ouvrir l'image : {image_path}")

    predict_kwargs = {"conf": confidence, "verbose": False}
    if imgsz:
        predict_kwargs["imgsz"] = imgsz
    result = model.predict(source=frame, **predict_kwargs)[0]
    annotated = result.plot()
    shark_detections = count_shark_detections(result, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    if display:
        cv2.imshow("Shark Sentinel - Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return shark_detections


def detect_on_image_dir(
    image_dir: Path, model: YOLO, output_dir: Path, confidence: float, display: bool, imgsz: int | None = None
) -> None:
    image_files = sorted(p for ext in IMAGE_EXTENSIONS for p in image_dir.glob(f"*{ext}"))
    if not image_files:
        raise RuntimeError(f"Aucune image trouvée dans : {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_shark_detections = 0
    print(f"Détection sur {len(image_files)} images...")

    for image_path in image_files:
        shark_detections = detect_on_image(
            image_path, model, output_dir / image_path.name, confidence, display, imgsz
        )
        total_shark_detections += shark_detections
        print(f"{image_path.name} : {shark_detections} détection(s) requin")

    print("\nAnalyse terminée.")
    print(f"Images annotées : {output_dir}")
    print(f"Images analysées : {len(image_files)}")
    print(f"Détections requin (total) : {total_shark_detections}")


def resolve_output_path(source_path: Path, output_arg: str, is_video: bool, is_dir: bool) -> Path:
    if output_arg:
        return Path(output_arg)
    if is_dir:
        return Path("runs/shark_sentinel/detections") / f"{source_path.name}_detected"
    if is_video:
        return Path("runs/shark_sentinel/detections/output.mp4")
    return Path("runs/shark_sentinel/detections") / f"{source_path.stem}_detected{source_path.suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Shark Sentinel - Detect sharks in an image, a folder of images, or a video"
    )
    parser.add_argument("--source", required=True, help="Chemin vers une image, un dossier d'images ou une vidéo")
    parser.add_argument("--model", required=True, help="Chemin vers le modèle entraîné .pt")
    parser.add_argument(
        "--output", default=None, help="Chemin de sortie (vidéo/image) ou dossier de sortie (dossier d'images)"
    )
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Taille d'image pour l'inférence (ex. 480, 320). Plus petit = plus rapide mais moins précis "
        "sur les petits requins (surtout en vue aérienne). Défaut : taille d'entraînement du modèle (640).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Vidéo uniquement : ne relance la détection qu'une frame sur N (accélère le traitement/l'affichage "
        "en réaffichant la dernière détection connue entre-temps). 1 = toutes les frames (défaut).",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Vidéo uniquement : active le système d'alerte à confirmation temporelle (voir --alert-window, "
        "--alert-min-hits, --alert-conf) au lieu de compter chaque détection indépendamment.",
    )
    parser.add_argument(
        "--alert-window",
        type=int,
        default=5,
        help="Nombre de frames analysées sur lesquelles évaluer une alerte (défaut : 5).",
    )
    parser.add_argument(
        "--alert-min-hits",
        type=int,
        default=3,
        help="Nombre minimum de frames positives dans --alert-window pour déclencher une alerte (défaut : 3).",
    )
    parser.add_argument(
        "--alert-conf",
        type=float,
        default=None,
        help="Confiance minimale pour qu'une détection compte dans le calcul d'alerte. "
        "Défaut : même valeur que --conf.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    model_path = Path(args.model)

    if not source_path.exists():
        raise FileNotFoundError(f"Source introuvable : {source_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    is_dir = source_path.is_dir()
    is_video = not is_dir and source_path.suffix.lower() in VIDEO_EXTENSIONS
    is_image = not is_dir and source_path.suffix.lower() in IMAGE_EXTENSIONS

    if not (is_dir or is_video or is_image):
        raise ValueError(
            f"Extension non reconnue : {source_path.suffix}. "
            f"Images : {sorted(IMAGE_EXTENSIONS)} — Vidéos : {sorted(VIDEO_EXTENSIONS)}"
        )

    model = YOLO(str(model_path))
    output_path = resolve_output_path(source_path, args.output, is_video, is_dir)

    if is_dir:
        detect_on_image_dir(source_path, model, output_path, args.conf, args.display, args.imgsz)
    elif is_video:
        alert_tracker = None
        if args.alert:
            alert_tracker = AlertTracker(
                window=args.alert_window,
                min_hits=args.alert_min_hits,
                alert_conf=args.alert_conf if args.alert_conf is not None else args.conf,
            )
        detect_on_video(
            source_path, model, output_path, args.conf, args.display, args.imgsz, args.frame_skip, alert_tracker
        )
    else:
        shark_detections = detect_on_image(source_path, model, output_path, args.conf, args.display, args.imgsz)
        print("\nAnalyse terminée.")
        print(f"Image de sortie : {output_path}")
        print(f"Détections requin : {shark_detections}")


if __name__ == "__main__":
    main()
