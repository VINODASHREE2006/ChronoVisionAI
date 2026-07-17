from ultralytics import YOLO

from config import *

model = YOLO(MODEL_PATH)

# Detect only people
results = model.predict(
    source=VIDEO_PATH,
    save=True,
    show=True,
    conf=0.3,
    classes=[0]   # Person class only
)

print("✅ Person detection completed!")