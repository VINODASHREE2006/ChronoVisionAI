import os
import sys

# Avoid OpenMP runtime conflicts on Windows/Anaconda setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
from ultralytics import YOLO

from src.timeline import TimelineGenerator
from src.utils import (
    calculate_distance,
    ensure_model,
    get_video_properties,
    inside_zone,
    person_label,
    scale_zone,
)
from src.zones import (
    CHECKOUT_ZONE,
    ENTRANCE_ZONE,
    EXIT_ZONE,
    QUEUE_ZONE,
    SHELF_ZONE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


class ActivityRecognizer:
    """Rule-based activity inference from movement and zone context."""

    RUN_THRESHOLD = 40
    WALK_THRESHOLD = 25
    SLOW_WALK_THRESHOLD = 10
    STABLE_FRAMES = 8
    WAITING_FRAMES = 45
    IDLE_FRAMES = 90
    LOST_FRAMES = 20

    def __init__(self, zones):
        self.zones = zones
        self.previous_positions = {}
        self.stable_counts = {}
        self.pending_activity = {}
        self.logged_activity = {}
        self.stationary_frames = {}
        self.shelf_frames = {}
        self.was_in_shelf = {}
        self.seen_tracks = set()
        self.missing_frames = {}

    def recognize(self, frame_number, track_id, center_x, center_y):
        track_id = int(track_id)

        if track_id not in self.previous_positions:
            self.previous_positions[track_id] = (center_x, center_y)
            self.stable_counts[track_id] = self.STABLE_FRAMES
            self.pending_activity[track_id] = "Entered"
            self.stationary_frames[track_id] = 0
            self.shelf_frames[track_id] = 0
            self.was_in_shelf[track_id] = False
            self.seen_tracks.add(track_id)
            return "Entered"

        prev_x, prev_y = self.previous_positions[track_id]
        distance = calculate_distance(prev_x, prev_y, center_x, center_y)
        self.previous_positions[track_id] = (center_x, center_y)

        if distance <= self.SLOW_WALK_THRESHOLD:
            self.stationary_frames[track_id] = (
                self.stationary_frames.get(track_id, 0) + 1
            )
        else:
            self.stationary_frames[track_id] = 0

        candidate = self._movement_activity(distance)

        if inside_zone(center_x, center_y, self.zones["checkout"]):
            candidate = "Checkout"
        elif inside_zone(center_x, center_y, self.zones["queue"]):
            candidate = "Queueing"
        elif inside_zone(center_x, center_y, self.zones["shelf"]):
            self.shelf_frames[track_id] = self.shelf_frames.get(track_id, 0) + 1
            if distance <= self.SLOW_WALK_THRESHOLD:
                if self.shelf_frames[track_id] >= self.STABLE_FRAMES:
                    candidate = "Shelf Interaction"
                if self.stationary_frames[track_id] >= self.WAITING_FRAMES // 2:
                    candidate = "Picking Product"
            else:
                candidate = "Slow Walking"
            self.was_in_shelf[track_id] = True
        elif self.was_in_shelf.get(track_id) and distance <= self.SLOW_WALK_THRESHOLD:
            candidate = "Returning Product"
            self.was_in_shelf[track_id] = False
            self.shelf_frames[track_id] = 0

        if (
            candidate in {"Standing", "Slow Walking"}
            and self.stationary_frames.get(track_id, 0) >= self.WAITING_FRAMES
        ):
            candidate = "Waiting"
        elif (
            candidate == "Standing"
            and self.stationary_frames.get(track_id, 0) >= self.IDLE_FRAMES
        ):
            candidate = "Idle"

        return self._stabilize(track_id, candidate)

    def register_logged_activity(self, track_id, activity):
        """Remember the last activity written to the timeline."""

        if activity:
            self.logged_activity[int(track_id)] = activity

    def update_missing_tracks(self, active_track_ids, frame_number):
        """Mark people as exited after they disappear from tracking."""

        events = []

        for track_id in list(self.seen_tracks):
            if track_id in active_track_ids:
                self.missing_frames[track_id] = 0
                continue

            self.missing_frames[track_id] = self.missing_frames.get(track_id, 0) + 1

            if (
                self.missing_frames[track_id] >= self.LOST_FRAMES
                and self.logged_activity.get(track_id) != "Exited"
            ):
                events.append((frame_number, track_id, "Exited"))
                self.logged_activity[track_id] = "Exited"
                self.pending_activity[track_id] = "Exited"
                self.seen_tracks.discard(track_id)

        return events

    def finalize_tracks(self, frame_number):
        """Mark remaining visible people as exited at video end."""

        events = []

        for track_id in list(self.seen_tracks):
            if self.logged_activity.get(track_id) != "Exited":
                events.append((frame_number, track_id, "Exited"))
                self.logged_activity[track_id] = "Exited"

        self.seen_tracks.clear()
        self.missing_frames.clear()
        return events

    def _movement_activity(self, distance):
        if distance > self.RUN_THRESHOLD:
            return "Running"
        if distance > self.WALK_THRESHOLD:
            return "Walking"
        if distance > self.SLOW_WALK_THRESHOLD:
            return "Slow Walking"
        return "Standing"

    def _stabilize(self, track_id, candidate):
        if candidate in {"Entered", "Exited"}:
            self.pending_activity[track_id] = candidate
            self.stable_counts[track_id] = self.STABLE_FRAMES
            return candidate

        if self.pending_activity.get(track_id) == candidate:
            self.stable_counts[track_id] = self.stable_counts.get(track_id, 0) + 1
        else:
            self.pending_activity[track_id] = candidate
            self.stable_counts[track_id] = 1

        if self.stable_counts[track_id] >= self.STABLE_FRAMES:
            return candidate

        return None


class VideoActivityAnalyzer:
    """End-to-end CCTV analysis pipeline."""

    MODEL_PATH = "models/yolov8n.pt"
    OUTPUT_DIR = "outputs"

    def __init__(self, video_path):
        self.video_path = video_path
        self.width, self.height, self.fps = get_video_properties(video_path)
        self.zones = self._build_zones()
        self.recognizer = ActivityRecognizer(self.zones)
        self.output_video_path = self._build_output_path()

    def _build_zones(self):
        return {
            "entrance": scale_zone(
                ENTRANCE_ZONE, VIDEO_WIDTH, VIDEO_HEIGHT, self.width, self.height
            ),
            "exit": scale_zone(
                EXIT_ZONE, VIDEO_WIDTH, VIDEO_HEIGHT, self.width, self.height
            ),
            "shelf": scale_zone(
                SHELF_ZONE, VIDEO_WIDTH, VIDEO_HEIGHT, self.width, self.height
            ),
            "checkout": scale_zone(
                CHECKOUT_ZONE, VIDEO_WIDTH, VIDEO_HEIGHT, self.width, self.height
            ),
            "queue": scale_zone(
                QUEUE_ZONE, VIDEO_WIDTH, VIDEO_HEIGHT, self.width, self.height
            ),
        }

    def _build_output_path(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(self.OUTPUT_DIR, f"annotated_{base_name}.mp4")

    def analyze(self):
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Unable to read video dimensions.")

        model_path = ensure_model(self.MODEL_PATH)
        model = YOLO(model_path)

        timeline = TimelineGenerator(fps=self.fps)
        latest_activity = {}
        writer = None

        try:
            writer = cv2.VideoWriter(
                self.output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Unable to create annotated video at {self.output_video_path}"
                )

            results = model.track(
                source=self.video_path,
                tracker="bytetrack.yaml",
                persist=True,
                stream=True,
                save=False,
                classes=[0],
                conf=0.3,
                verbose=False,
            )

            frame_number = 0

            for result in results:
                frame_number += 1
                frame = result.orig_img.copy()
                active_track_ids = set()

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)

                    for box, track_id in zip(boxes, ids):
                        active_track_ids.add(int(track_id))

                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        activity = self.recognizer.recognize(
                            frame_number,
                            track_id,
                            center_x,
                            center_y,
                        )

                        if activity:
                            timeline.add_event(frame_number, track_id, activity)
                            latest_activity[track_id] = activity
                            self.recognizer.register_logged_activity(track_id, activity)

                        label = latest_activity.get(track_id, "Detected")
                        self._draw_annotation(frame, x1, y1, x2, y2, track_id, label)

                for exit_frame, track_id, activity in self.recognizer.update_missing_tracks(
                    active_track_ids,
                    frame_number,
                ):
                    timeline.add_event(exit_frame, track_id, activity)
                    latest_activity.pop(track_id, None)

                writer.write(frame)

            for exit_frame, track_id, activity in self.recognizer.finalize_tracks(
                frame_number
            ):
                timeline.add_event(exit_frame, track_id, activity)

        finally:
            if writer is not None:
                writer.release()
            timeline.close()

        return self.output_video_path

    @staticmethod
    def _draw_annotation(frame, x1, y1, x2, y2, track_id, activity):
        color = (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{person_label(track_id)} | {activity}"
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        text_w, text_h = text_size
        top = max(y1 - text_h - 10, 0)

        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + text_w + 8, top + text_h + 8),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.activity <video_path>", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        analyzer = VideoActivityAnalyzer(video_path)
        output_path = analyzer.analyze()
        print("Timeline generated successfully.")
        print(f"Annotated video saved to: {output_path}")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
