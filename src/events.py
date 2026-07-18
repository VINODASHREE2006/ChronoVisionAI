import csv
from ultralytics import YOLO

from src.config import *

model = YOLO(MODEL_PATH)

# Store tracked persons
tracked_people = set()

# Create CSV file
with open("data/timeline.csv", "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(["Frame", "Person_ID", "Event"])

    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        classes=[0]
    )

    frame_no = 0

    for result in results:

        frame_no += 1

        if result.boxes.id is None:
            continue

        ids = result.boxes.id.cpu().numpy().astype(int)

        for person_id in ids:

            if person_id not in tracked_people:

                tracked_people.add(person_id)

                writer.writerow([frame_no, person_id, "Entered"])

print("Timeline Created Successfully!")