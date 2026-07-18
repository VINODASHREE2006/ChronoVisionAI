import csv
import os

from src.utils import frame_to_time, person_label


class TimelineGenerator:
    """Write deduplicated activity events to timeline.csv."""

    COLUMNS = ["Timestamp", "Person", "Activity", "Confidence", "Reason"]

    def __init__(self, fps=30, output_path="data/timeline.csv"):
        self.fps = fps if fps > 0 else 30
        self.file_path = output_path
        self.last_activity = {}
        self._file = None
        self._writer = None
        self._open_file()

    def _open_file(self):
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

        self._file = open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._file.flush()

    def add_event(self, timestamp, person_id, activity, confidence="", reason=""):
        """Add an event only when the activity changes for a person."""

        if not activity:
            return

        try:
            person_id = int(person_id)
        except (TypeError, ValueError):
            return

        if not timestamp:
            return

        if (
            person_id in self.last_activity
            and self.last_activity[person_id] == activity
        ):
            return

        self.last_activity[person_id] = activity

        self._writer.writerow([
            timestamp,
            person_label(person_id),
            str(activity).strip(),
            str(confidence),
            str(reason).strip(),
        ])
        self._file.flush()

    def close(self):
        """Close the CSV file handle safely."""

        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
