from pathlib import Path
from ultralytics import YOLO


class EmergencyCarDetector:

    def __init__(self, weights: str = "yolov8n.pt"):
        self.weights = weights
        self.model   = YOLO(weights)

    def detect(
        self,
        source,
        conf:    float = 0.25,
        iou:     float = 0.45,
        imgsz:   int   = 640,
        save:    bool  = False,
        show:    bool  = False,
        device:  str   = "",
    ):
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            show=show,
            device=device if device else None,
            verbose=False,
        )
        return results

    def is_emergency(self, source, conf: float = 0.25) -> dict:
        results = self.detect(source, conf=conf)

        all_classes    = []
        max_confidence = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf_val = float(box.conf[0])
                all_classes.append(cls_name)
                if conf_val > max_confidence:
                    max_confidence = conf_val

        return {
            "emergency":  len(all_classes) > 0,
            "detections": len(all_classes),
            "confidence": round(max_confidence, 4),
            "classes":    all_classes,
        }

    def export(self, format: str = "onnx"):
        """
        Export model to a deployment format.
        Formats: onnx, torchscript, openvino, tflite, coreml
        """
        path = self.model.export(format=format)
        print(f"Exported to: {path}")
        return path