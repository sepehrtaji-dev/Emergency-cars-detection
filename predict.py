import argparse
import sys
from pathlib import Path
from model import EmergencyCarDetector

WEIGHTS_DEFAULT = "runs/detect/emergency_car/weights/best.pt"

def run(
    source:  str,
    weights: str  = WEIGHTS_DEFAULT,
    conf:    float = 0.25,
    iou:     float = 0.45,
    imgsz:   int   = 640,
    save:    bool  = False,
    show:    bool  = False,
    device:  str   = "",
):
    if not Path(weights).exists():
        print(f"[ERROR] Weights not found: {weights}")
        print("Train first: python train.py")
        sys.exit(1)

    print(f"Loading model: {weights}")
    detector = EmergencyCarDetector(weights=weights)

    print(f"Running on: {source}")
    results = detector.detect(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save,
        show=show,
        device=device,
    )

    print("\n── Results ─────────────────────────────────")
    for i, result in enumerate(results):
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print(f"  [{i}] No detections")
            continue

        print(f"  [{i}] {result.path}")
        for box in boxes:
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])
            xyxy     = box.xyxy[0].tolist()
            print(
                f"       {cls_name:<20} conf={conf_val:.3f}  "
                f"box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]"
            )

    total = sum(
        len(r.boxes) for r in results if r.boxes is not None
    )
    print(f"\nTotal detections: {total}")

    if save:
        save_dir = results[0].save_dir if results else "runs/detect/predict"
        print(f"Results saved to: {save_dir}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency Car Detector Inference")

    parser.add_argument("--source",  required=True,                    help="image/video/folder/0 for webcam")
    parser.add_argument("--weights", default=WEIGHTS_DEFAULT,          help="model weights path")
    parser.add_argument("--conf",    default=0.25, type=float,         help="confidence threshold")
    parser.add_argument("--iou",     default=0.45, type=float,         help="NMS IoU threshold")
    parser.add_argument("--imgsz",   default=640,  type=int,           help="inference image size")
    parser.add_argument("--save",    action="store_true",              help="save annotated results")
    parser.add_argument("--show",    action="store_true",              help="display results")
    parser.add_argument("--device",  default="",                       help="cpu / 0 / 0,1")

    args = parser.parse_args()
    run(**vars(args))