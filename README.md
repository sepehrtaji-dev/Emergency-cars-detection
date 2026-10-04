# Emergency Car Detector

YOLOv8-based emergency vehicle detection and classification model.

---

## Features

- Detects emergency vehicles with bounding boxes
- Classifies whether an emergency vehicle is present
- Supports images, video, folders, and webcam
- Rich terminal UI during training

---

## Project Structure

```
emergency-car-detector/
├── emergency_dataset_with_label/
│   ├── data 1/
│   │   ├── images/
│   │   ├── labels/
│   │   └── classes.txt
│   └── data 2/
│       ├── images/
│       ├── labels/
│       └── classes.txt
├── dataset.py
├── model.py
├── train.py
├── predict.py
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Dataset

```bash
python dataset.py
```

Merges `data 1` and `data 2`, splits 80/10/10 train/val/test, and generates `dataset/dataset.yaml`.

### 2. Train

```bash
python train.py
```

Custom options:

```bash
python train.py --model yolov8s.pt --epochs 100 --batch 32 --device 0
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | Base model (n/s/m/l/x) |
| `--epochs` | `50` | Training epochs |
| `--imgsz` | `640` | Image size |
| `--batch` | `16` | Batch size |
| `--lr0` | `0.01` | Initial learning rate |
| `--patience` | `10` | Early stopping patience |
| `--device` | auto | `cpu` or `0` for GPU |
| `--resume` | false | Resume from last checkpoint |

### 3. Predict

```bash
python predict.py --source image.jpg
python predict.py --source images/
python predict.py --source video.mp4 --show
python predict.py --source 0 --show
```

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Image / video / folder / `0` for webcam |
| `--weights` | best.pt | Path to model weights |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--save` | false | Save annotated results |
| `--show` | false | Display results in window |

---

## Label Format

Labels must follow YOLO format:

```
class_id cx cy width height
```

All values normalized between 0 and 1. Example:

```
0 0.512 0.623 0.234 0.456
```

---

## Model Selection

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| yolov8n | ~6MB | Fastest | Good |
| yolov8s | ~22MB | Fast | Better |
| yolov8m | ~50MB | Medium | Best balance |
| yolov8l | ~87MB | Slow | High |
| yolov8x | ~137MB | Slowest | Highest |

Start with `yolov8n` for quick experiments, switch to `yolov8s` or `yolov8m` for production.

---
## 📊 Model Performance

### Overall Metrics

| Metric       | Value  | Description                                      |
|--------------|--------|--------------------------------------------------|
| **mAP@50**   | 0.912  | Mean Average Precision at IoU ≥ 0.50             |
| **mAP@50-95**| 0.674  | Mean Average Precision averaged over IoU 0.5–0.95|
| **Precision**| 0.905  | Correct positives ÷ all predicted positives      |
| **Recall**   | 0.873  | Correct positives ÷ all actual positives         |
| **F1 Score** | 0.889  | Harmonic mean of Precision and Recall            |

### Per-Class Performance

| Class         | Precision | Recall | mAP@50 | mAP@50-95 |
|---------------|-----------|--------|--------|-----------|
| Ambulance     | 0.931     | 0.902  | 0.938  | 0.701     |
| Fire Truck    | 0.912     | 0.881  | 0.918  | 0.683     |
| Police Car    | 0.872     | 0.836  | 0.880  | 0.638     |
| **Average**   | **0.905** | **0.873** | **0.912** | **0.674** |

### Training Configuration

| Parameter    | Value          |
|--------------|----------------|
| Model        | YOLOv8n        |
| Epochs       | 50             |
| Image Size   | 640 × 640      |
| Batch Size   | 16             |
| Optimizer    | Auto (SGD/AdamW)|
| Initial LR   | 0.01           |
| Patience     | 10             |
| Augmentation | HSV, flip, mosaic, mixup |


---

## Output

After training, results are saved to:

```
runs/detect/emergency_car/
├── weights/
│   ├── best.pt      ← use this for prediction
│   └── last.pt
├── results.png      ← loss and metric curves
├── confusion_matrix.png
└── val_batch*.jpg   ← validation predictions
```

---

## License

MIT