from ultralytics import YOLO
from src.timeline import TimelineGenerator
from src.utils import calculate_distance

# ----------------------------------------
# Load YOLO Model
# ----------------------------------------

model = YOLO("models/yolov8n.pt")

# ----------------------------------------
# Timeline Generator
# ----------------------------------------

timeline = TimelineGenerator(fps=30)

# ----------------------------------------
# Previous Positions
# ----------------------------------------

previous_positions = {}

# ----------------------------------------
# Track Video
# ----------------------------------------

results = model.track(
    source="videos/test.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    save=True,
    classes=[0],
    conf=0.3
)

frame_number = 0

# ----------------------------------------
# Process Frames
# ----------------------------------------

for result in results:

    frame_number += 1

    if result.boxes.id is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy()

    for box, track_id in zip(boxes, ids):

        track_id = int(track_id)

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # ----------------------------------
        # First Detection
        # ----------------------------------

        if track_id not in previous_positions:

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            timeline.add_event(
                frame_number,
                track_id,
                "Entered"
            )

            continue

        prev_x, prev_y = previous_positions[track_id]

        distance = calculate_distance(
            prev_x,
            prev_y,
            center_x,
            center_y
        )

        previous_positions[track_id] = (
            center_x,
            center_y
        )

        # ----------------------------------
        # Activity Recognition
        # ----------------------------------

        if distance > 25:

            activity = "Walking"

        elif distance > 10:

            activity = "Slow Walking"

        else:

            activity = "Standing"

        timeline.add_event(
            frame_number,
            track_id,
            activity
        )

timeline.close()

print("✅ Timeline Generated Successfully!")