# app.py
import os
import sys
import glob
from pathlib import Path

from PySide6.QtCore import (
    Qt, QTimer, QPointF, QRectF, Signal, QThread, QSize
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QBrush, QPen, QFont,
    QLinearGradient, QRadialGradient, QFontDatabase, QPainterPath,
    QDragEnterEvent, QDropEvent
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QFrame, QGraphicsDropShadowEffect, QSizePolicy,
    QScrollArea, QGraphicsBlurEffect
)

from ultralytics import YOLO
from PIL import Image


# ----------------------------------------------------------------------------
#  Locate trained weights
# ----------------------------------------------------------------------------
def find_weights() -> str:
    candidates = [
        "runs/detect/runs/detect/emergency_car-4/weights/best.pt",
        "runs/detect/runs/detect/emergency_car-3/weights/best.pt",
        "runs/detect/emergency_car/weights/best.pt",
        "runs/detect/runs/detect/emergency_car/weights/best.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    matches = sorted(
        glob.glob("runs/**/weights/best.pt", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    return matches[0] if matches else "yolov8n.pt"


WEIGHTS = find_weights()
print(f"[INFO] Loading model: {WEIGHTS}")
MODEL = YOLO(WEIGHTS)


# ----------------------------------------------------------------------------
#  Background widget: dark base + colorful animated blobs
# ----------------------------------------------------------------------------
class AuroraBackground(QWidget):
    """Animated gradient blobs to give the glassmorphism look something to blur."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.t = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)  # ~25 fps

    def _tick(self):
        self.t = (self.t + 0.006) % 1.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Base dark gradient
        base = QLinearGradient(0, 0, self.width(), self.height())
        base.setColorAt(0.0, QColor("#05060f"))
        base.setColorAt(1.0, QColor("#0a0820"))
        p.fillRect(self.rect(), base)

        import math

        def blob(cx, cy, r, color_a, color_b):
            g = QRadialGradient(cx, cy, r)
            g.setColorAt(0.0, color_a)
            g.setColorAt(1.0, color_b)
            p.setBrush(QBrush(g))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), r, r)

        w, h = self.width(), self.height()
        phase = self.t * 2 * math.pi

        # Purple blob (top-left)
        blob(
            w * (0.20 + 0.05 * math.sin(phase)),
            h * (0.20 + 0.05 * math.cos(phase)),
            min(w, h) * 0.55,
            QColor(124, 92, 255, 200),
            QColor(124, 92, 255, 0),
        )

        # Teal blob (bottom-right)
        blob(
            w * (0.80 + 0.06 * math.cos(phase * 0.9 + 1)),
            h * (0.80 + 0.05 * math.sin(phase * 1.1 + 2)),
            min(w, h) * 0.50,
            QColor(0, 224, 198, 180),
            QColor(0, 224, 198, 0),
        )

        # Pink blob (middle)
        blob(
            w * (0.60 + 0.08 * math.sin(phase * 0.7 + 3)),
            h * (0.45 + 0.07 * math.cos(phase * 0.8 + 1)),
            min(w, h) * 0.35,
            QColor(255, 77, 109, 140),
            QColor(255, 77, 109, 0),
        )


# ----------------------------------------------------------------------------
#  Glass panel: translucent rounded card with subtle inner highlight
# ----------------------------------------------------------------------------
class GlassPanel(QFrame):
    def __init__(self, radius: int = 24, parent=None):
        super().__init__(parent)
        self.radius = radius
        self.setAttribute(Qt.WA_StyledBackground, False)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        r = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(QRectF(r), self.radius, self.radius)

        # Translucent fill
        p.fillPath(path, QBrush(QColor(255, 255, 255, 22)))

        # Top highlight
        highlight = QLinearGradient(0, r.top(), 0, r.top() + 60)
        highlight.setColorAt(0.0, QColor(255, 255, 255, 60))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(path, QBrush(highlight))

        # Border
        pen = QPen(QColor(255, 255, 255, 55), 1.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)


# ----------------------------------------------------------------------------
#  Drop zone
# ----------------------------------------------------------------------------
class DropZone(QFrame):
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(260)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        self.icon = QLabel("⬆")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("""
            QLabel {
                font-size: 34px;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #7c5cff, stop:1 #00e0c6);
                border-radius: 34px;
                padding: 12px;
            }
        """)
        self.icon.setFixedSize(68, 68)

        icon_wrap = QWidget()
        icon_wrap_layout = QVBoxLayout(icon_wrap)
        icon_wrap_layout.setContentsMargins(0, 0, 0, 0)
        icon_wrap_layout.setAlignment(Qt.AlignCenter)
        icon_wrap_layout.addWidget(self.icon)

        self.title = QLabel("Drop an image here")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #eef2ff; font-size: 18px; font-weight: 700;")

        self.sub = QLabel("or click to browse  ·  JPG, PNG, WEBP")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setStyleSheet("color: #a5b0d4; font-size: 13px;")

        layout.addWidget(icon_wrap, alignment=Qt.AlignHCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.sub)

    # --- drag & drop ---
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self.update()

    def dragLeaveEvent(self, e):
        self._hover = False
        self.update()

    def dropEvent(self, e: QDropEvent):
        self._hover = False
        self.update()
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")):
                self.fileDropped.emit(path)
                break

    def mousePressEvent(self, e):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if path:
            self.fileDropped.emit(path)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)

        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 20, 20)

        fill = QColor(255, 255, 255, 12)
        if self._hover:
            fill = QColor(124, 92, 255, 40)
        p.fillPath(path, QBrush(fill))

        # Dashed border
        pen = QPen(QColor(255, 255, 255, 90 if not self._hover else 180), 1.6, Qt.DashLine)
        pen.setDashPattern([6, 5])
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)


# ----------------------------------------------------------------------------
#  Result area: image with bounding boxes drawn on top
# ----------------------------------------------------------------------------
class ResultView(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self._pixmap = None
        self._boxes = []      # list of dict: x1,y1,x2,y2,cls,conf
        self._img_size = (1, 1)

    def set_result(self, pil_image, boxes):
        self._boxes = boxes
        self._img_size = pil_image.size  # (w, h)

        # PIL -> QPixmap
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(data, pil_image.width, pil_image.height,
                      pil_image.width * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def clear(self):
        self._pixmap = None
        self._boxes = []
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        r = self.rect()

        if self._pixmap is None:
            # Placeholder
            p.setPen(QColor(180, 190, 220, 120))
            f = QFont()
            f.setPointSize(12)
            p.setFont(f)
            p.drawText(r, Qt.AlignCenter, "No image analyzed yet")
            return

        # Fit image inside widget preserving aspect ratio
        pix = self._pixmap
        scaled = pix.scaled(r.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        ox = (r.width()  - scaled.width())  // 2
        oy = (r.height() - scaled.height()) // 2

        # Rounded clip for the image
        path = QPainterPath()
        path.addRoundedRect(QRectF(ox, oy, scaled.width(), scaled.height()), 16, 16)
        p.setClipPath(path)
        p.drawPixmap(ox, oy, scaled)
        p.setClipping(False)

        # Scale factor from original image coords to displayed size
        sx = scaled.width()  / self._img_size[0]
        sy = scaled.height() / self._img_size[1]

        # Draw boxes
        for b in self._boxes:
            x1 = ox + b["x1"] * sx
            y1 = oy + b["y1"] * sy
            x2 = ox + b["x2"] * sx
            y2 = oy + b["y2"] * sy

            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            # Glow
            glow = QColor(255, 77, 109, 90)
            p.setPen(QPen(glow, 8))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect, 8, 8)

            # Main outline
            p.setPen(QPen(QColor("#ff4d6d"), 2.6))
            p.drawRoundedRect(rect, 8, 8)

            # Label
            label = f'{b["cls"]}  {b["conf"]*100:.1f}%'
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 16
            th = fm.height() + 8
            lx = x1
            ly = max(oy, y1 - th - 4)

            label_bg = QRectF(lx, ly, tw, th)
            p.setBrush(QBrush(QColor(255, 77, 109, 220)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(label_bg, 6, 6)

            p.setPen(QPen(QColor("#ffffff")))
            f = QFont()
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            p.drawText(label_bg.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, label)


# ----------------------------------------------------------------------------
#  Detection worker thread
# ----------------------------------------------------------------------------
class DetectWorker(QThread):
    finished = Signal(object, list)   # pil_image, boxes
    error    = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            img = Image.open(self.path).convert("RGB")
            results = MODEL.predict(source=img, conf=0.25, iou=0.45, verbose=False)
            r = results[0]

            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cls": r.names[cls_id],
                        "conf": conf,
                    })
            self.finished.emit(img, boxes)
        except Exception as e:
            self.error.emit(str(e))


# ----------------------------------------------------------------------------
#  Status pill
# ----------------------------------------------------------------------------
class StatusPill(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self._mode = "idle"   # idle | emergency | safe | busy
        self._text = "Waiting for image…"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(14)

        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)
        self.dot.setStyleSheet("border-radius: 6px; background: #a5b0d4;")

        self.label = QLabel(self._text)
        self.label.setStyleSheet("color: #eef2ff; font-size: 14px; font-weight: 700;")

        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        layout.addStretch()

    def set_state(self, mode: str, text: str):
        self._mode = mode
        self._text = text
        self.label.setText(text)
        colors = {
            "idle":      ("#a5b0d4", "background: rgba(255,255,255,0.06);  border: 1px solid rgba(255,255,255,0.12);"),
            "busy":      ("#7c5cff", "background: rgba(124,92,255,0.16);  border: 1px solid rgba(124,92,255,0.35);"),
            "emergency": ("#ff4d6d", "background: rgba(255,77,109,0.16);  border: 1px solid rgba(255,77,109,0.4);"),
            "safe":      ("#2ee6a6", "background: rgba(46,230,166,0.12);  border: 1px solid rgba(46,230,166,0.35);"),
        }
        dot_color, frame_style = colors.get(mode, colors["idle"])
        self.dot.setStyleSheet(f"border-radius: 6px; background: {dot_color};")
        self.setStyleSheet(f"StatusPill {{ border-radius: 14px; {frame_style} }}")


# ----------------------------------------------------------------------------
#  Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emergency Car Detector")
        self.resize(1120, 860)

        # Background as central widget
        self.bg = AuroraBackground(self)
        self.setCentralWidget(self.bg)

        # Root layout on top of the background
        root = QVBoxLayout(self.bg)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(22)

        # ---- Header ----
        header = QVBoxLayout()
        header.setSpacing(4)

        self.h1 = QLabel("Emergency Car Detector")
        self.h1.setAlignment(Qt.AlignCenter)
        self.h1.setStyleSheet("""
            QLabel {
                font-size: 34px;
                font-weight: 800;
                color: white;
            }
        """)

        self.h2 = QLabel("Upload an image — emergency vehicles will be highlighted.")
        self.h2.setAlignment(Qt.AlignCenter)
        self.h2.setStyleSheet("color: #a5b0d4; font-size: 13px;")

        header.addWidget(self.h1)
        header.addWidget(self.h2)
        root.addLayout(header)

        # ---- Main glass card ----
        self.card = GlassPanel(radius=26)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(18)

        # Drop zone
        self.drop = DropZone()
        self.drop.fileDropped.connect(self.on_file)
        card_layout.addWidget(self.drop)

        # Result view (hidden initially)
        self.result = ResultView()
        self.result.setVisible(False)
        card_layout.addWidget(self.result)

        # Status pill
        self.status = StatusPill()
        self.status.setVisible(False)
        card_layout.addWidget(self.status)

        # Detections container
        self.detections_box = QVBoxLayout()
        self.detections_box.setSpacing(8)
        card_layout.addLayout(self.detections_box)

        # Reset button
        self.btn_row = QHBoxLayout()
        self.btn_row.addStretch()
        self.reset_btn = QPushButton("Analyze another image")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setFixedHeight(44)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                color: white;
                font-weight: 700;
                font-size: 13px;
                padding: 0 22px;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.18);
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 rgba(124,92,255,0.85),
                                            stop:1 rgba(162,123,255,0.85));
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 rgba(124,92,255,1.0),
                                            stop:1 rgba(162,123,255,1.0));
            }
            QPushButton:pressed { padding-top: 2px; }
        """)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setVisible(False)
        self.btn_row.addWidget(self.reset_btn)
        card_layout.addLayout(self.btn_row)

        root.addWidget(self.card, 1)

        # Footer
        self.footer = QLabel("Powered by YOLOv8  ·  Emergency Car Detector")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setStyleSheet("color: #6b7aa1; font-size: 11px;")
        root.addWidget(self.footer)

        self.worker = None

    # ------------------------------------------------------------------
    def on_file(self, path: str):
        # Hide drop zone, show result
        self.drop.setVisible(False)
        self.result.setVisible(True)
        self.status.setVisible(True)
        self.reset_btn.setVisible(True)

        self.clear_detections()
        self.result.clear()
        self.status.set_state("busy", "Analyzing…")

        self.worker = DetectWorker(path)
        self.worker.finished.connect(self.on_detected)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_detected(self, img, boxes):
        self.result.set_result(img, boxes)

        if boxes:
            self.status.set_state(
                "emergency",
                f"🚨 EMERGENCY detected — {len(boxes)} object{'s' if len(boxes) != 1 else ''}",
            )
        else:
            self.status.set_state("safe", "✓ NON-EMERGENCY — no emergency vehicles found")

        self.render_detections(boxes)

    def on_error(self, msg: str):
        self.status.set_state("emergency", f"Error: {msg}")

    def render_detections(self, boxes):
        self.clear_detections()
        for b in sorted(boxes, key=lambda x: -x["conf"]):
            row = QFrame()
            row.setStyleSheet("""
                QFrame {
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 12px;
                }
            """)
            row.setFixedHeight(46)

            h = QHBoxLayout(row)
            h.setContentsMargins(18, 0, 18, 0)

            name = QLabel(b["cls"])
            name.setStyleSheet("color: #eef2ff; font-size: 13px; font-weight: 700; text-transform: capitalize;")

            conf = QLabel(f'{b["conf"]*100:.1f}%')
            conf.setStyleSheet("color: #00e0c6; font-size: 13px; font-weight: 700;")

            h.addWidget(name)
            h.addStretch()
            h.addWidget(conf)

            self.detections_box.addWidget(row)

    def clear_detections(self):
        while self.detections_box.count():
            item = self.detections_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def reset(self):
        self.result.clear()
        self.result.setVisible(False)
        self.status.setVisible(False)
        self.reset_btn.setVisible(False)
        self.clear_detections()
        self.drop.setVisible(True)


# ----------------------------------------------------------------------------
#  Entry point
# ----------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)

    # A nicer default font if available
    for name in ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue"):
        if name in QFontDatabase.families():
            app.setFont(QFont(name, 10))
            break

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()