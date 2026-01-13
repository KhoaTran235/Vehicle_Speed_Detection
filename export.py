from ultralytics import YOLO
import tensorrt as trt

print(trt.__version__)

model = YOLO("models/pruned_yolov8n.pt")
model.export(
    format="engine",
    imgsz=640,
    batch=8,        # max batch
    dynamic=True,
    half=True,
    nms=True,
    simplify=True
)