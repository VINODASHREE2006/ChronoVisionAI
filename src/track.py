from ultralytics import YOLO

from src.config import *

model = YOLO(MODEL_PATH)

# Track people
results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    show=False,
    save=True,
    classes=[0],
    conf=0.3,
)

print("✅ Tracking completed!")