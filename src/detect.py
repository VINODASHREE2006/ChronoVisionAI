from ultralytics import YOLO

from src.config import *

model = YOLO(MODEL_PATH)

# Detect only people
results = model.predict(
    source=VIDEO_PATH,
    save=True,
    show=False,
    conf=0.3,
    classes=[0],
)

print("✅ Person detection completed!")