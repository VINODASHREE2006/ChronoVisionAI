import csv
import os

from src.utils import frame_to_time


class TimelineGenerator:

    def __init__(self, fps=30):

        self.fps = fps

        os.makedirs("data", exist_ok=True)

        self.file_path = "data/timeline.csv"

        self.file = open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.file)

        # CSV Header
        self.writer.writerow([
            "Time",
            "Person ID",
            "Activity"
        ])

        # Store last activity of each person
        self.last_activity = {}

    def add_event(self, frame, person_id, activity):
        """
        Add an event only if the activity changed.
        """

        if (
            person_id in self.last_activity
            and self.last_activity[person_id] == activity
        ):
            return

        self.last_activity[person_id] = activity

        timestamp = frame_to_time(frame, self.fps)

        self.writer.writerow([
            timestamp,
            person_id,
            activity
        ])

        # Save immediately
        self.file.flush()

    def close(self):

        if not self.file.closed:
            self.file.close()