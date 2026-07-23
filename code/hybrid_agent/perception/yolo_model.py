from ultralytics import YOLO
import torch

from utils.timer import Timer


class YoloModel:
    """
    Lightweight wrapper around a YOLO detector for anatomical object detection.
    """

    def __init__(self, weights_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        print(f"YOLO model loaded on device: {self.device}")

        if self.device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def detect(self, image_path):
        """
        Run YOLO inference on one image and return normalized detection dictionaries.
        """

        timer = Timer()
        timer.start("YOLO inference")

        results = self.model(
            image_path,
            conf=1e-6,
            iou=0.5,
            imgsz=512,
            verbose=False,
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = self.model.names[cls_id]

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name.lower(),
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                    ],
                })

        timer.stop("YOLO inference")

        return detections

    def get_class_names(self):
        """
        Return class names known to the YOLO model.
        """
        return list(self.model.names.values())