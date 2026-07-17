from ultralytics import YOLO

from config import *

model = YOLO(MODEL_PATH)

# Track people
results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    show=True,
    save=True,
    classes=[0],      # Person only
    conf=0.3
)

print("✅ Tracking completed!")