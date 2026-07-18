import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import cv2
import numpy as np
from ultralytics import YOLO

from src import config
from src.pose import PoseEstimator
from src.timestamp_ocr import CCTVTimestampExtractor
from src.timeline import TimelineGenerator
from src.utils import (
    box_foot_point,
    box_height,
    calculate_distance,
    ensure_model,
    filter_detections,
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


class PersonIdentityManager:
    """Map raw tracker IDs to stable sequential Person labels."""

    def __init__(self, reid_distance_ratio, lost_frames):
        self.reid_distance_ratio = reid_distance_ratio
        self.lost_frames = lost_frames
        self.raw_to_display = {}
        self.next_display_id = 1
        self.lost_tracks = {}

    def assign(self, raw_id, foot_x, foot_y, box_h, frame_number, confirmed):
        raw_id = int(raw_id)

        if raw_id in self.raw_to_display:
            display_id = self.raw_to_display[raw_id]
            self.lost_tracks.pop(display_id, None)
            return display_id

        if not confirmed:
            return None

        display_id = self._match_lost_track(foot_x, foot_y, box_h, frame_number)
        if display_id is None:
            display_id = self.next_display_id
            self.next_display_id += 1

        self.raw_to_display[raw_id] = display_id
        return display_id

    def purge_raw_id(self, display_id):
        # Remove mapping so if tracker revives the raw_id, it gets a new display_id
        keys_to_delete = [r_id for r_id, d_id in self.raw_to_display.items() if d_id == display_id]
        for k in keys_to_delete:
            del self.raw_to_display[k]

    def mark_missing(self, display_id, foot_x, foot_y, box_h, frame_number):
        if display_id is None:
            return
        self.lost_tracks[display_id] = {
            "foot_x": foot_x,
            "foot_y": foot_y,
            "box_h": box_h,
            "frame": frame_number,
        }

    def _match_lost_track(self, foot_x, foot_y, box_h, frame_number):
        best_id = None
        best_distance = float("inf")

        for display_id, info in list(self.lost_tracks.items()):
            if frame_number - info["frame"] > self.lost_frames:
                del self.lost_tracks[display_id]
                continue

            threshold = self.reid_distance_ratio * max(box_h, info["box_h"])
            distance = calculate_distance(
                info["foot_x"],
                info["foot_y"],
                foot_x,
                foot_y,
            )

            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_id = display_id

        if best_id is not None:
            del self.lost_tracks[best_id]

        return best_id

    @property
    def confirmed_person_count(self):
        return len(set(self.raw_to_display.values()))


class TrackValidator:
    """Confirm tracks before they become timeline persons."""

    def __init__(self, min_confirm_frames, min_track_lifetime, lost_frames):
        self.min_confirm_frames = min_confirm_frames
        self.min_track_lifetime = min_track_lifetime
        self.lost_frames = lost_frames
        self.frame_counts = {}
        self.missing_counts = {}
        self.confirmed = set()
        self.valid_lifetime = set()

    def update(self, raw_id, present):
        raw_id = int(raw_id)

        if present:
            self.frame_counts[raw_id] = self.frame_counts.get(raw_id, 0) + 1
            self.missing_counts[raw_id] = 0

            if self.frame_counts[raw_id] >= self.min_confirm_frames:
                self.confirmed.add(raw_id)
        else:
            self.missing_counts[raw_id] = self.missing_counts.get(raw_id, 0) + 1

            if self.missing_counts[raw_id] >= self.lost_frames:
                if self.frame_counts.get(raw_id, 0) >= self.min_track_lifetime:
                    self.valid_lifetime.add(raw_id)
                self.confirmed.discard(raw_id)

    def is_confirmed(self, raw_id):
        return int(raw_id) in self.confirmed

    def is_valid(self, raw_id):
        raw_id = int(raw_id)
        if raw_id in self.valid_lifetime:
            return True
        if raw_id in self.confirmed:
            return True
        return self.frame_counts.get(raw_id, 0) >= self.min_confirm_frames


class ActivityRecognizer:
    """Rule-based activity inference with zone dwell and debouncing."""

    def __init__(self, zones, fps):
        self.zones = zones
        self.fps = max(fps, 1)
        self.previous_positions = {}
        self.stable_counts = {}
        self.pending_activity = {}
        self.logged_activity = {}
        self.stationary_frames = {}
        self.zone_dwell = {}
        self.shelf_dwell = {}
        self.was_in_shelf = {}
        self.seen_display_ids = set()
        self.missing_frames = {}
        self.smoothed_move = {}
        self.candidate_history = {}

    def recognize(
        self,
        frame_number,
        display_id,
        foot_x,
        foot_y,
        movement_distance,
        box_h,
        is_picking=False,
    ):
        display_id = int(display_id)

        if display_id not in self.previous_positions:
            self.previous_positions[display_id] = (foot_x, foot_y)
            self.stable_counts[display_id] = config.STABLE_FRAMES
            self.pending_activity[display_id] = "Entered"
            self.logged_activity.pop(display_id, None)
            self.stationary_frames[display_id] = 0
            self.zone_dwell[display_id] = {}
            self.shelf_dwell[display_id] = 0
            self.was_in_shelf[display_id] = False
            self.seen_display_ids.add(display_id)
            self.smoothed_move[display_id] = 0.0
            self.candidate_history[display_id] = []
            return "Entered"

        current_norm_move = movement_distance / max(box_h, 1.0)
        smoothed = self.smoothed_move.get(display_id, current_norm_move)
        alpha = 0.05
        norm_move = alpha * current_norm_move + (1 - alpha) * smoothed
        self.smoothed_move[display_id] = norm_move

        if norm_move <= config.SLOW_WALK_RATIO:
            self.stationary_frames[display_id] = (
                self.stationary_frames.get(display_id, 0) + 1
            )
        else:
            self.stationary_frames[display_id] = 0

        candidate = self._movement_activity(norm_move)
        candidate = self._apply_zone_logic(
            display_id,
            foot_x,
            foot_y,
            norm_move,
            candidate,
            is_picking,
        )

        return self._stabilize(display_id, candidate)

    def register_logged_activity(self, display_id, activity):
        if activity:
            self.logged_activity[int(display_id)] = activity

    def update_missing_tracks(self, active_display_ids, frame_number):
        events = []

        for display_id in list(self.seen_display_ids):
            if display_id in active_display_ids:
                self.missing_frames[display_id] = 0
                continue

            self.missing_frames[display_id] = (
                self.missing_frames.get(display_id, 0) + 1
            )

            if (
                self.missing_frames[display_id] >= config.LOST_FRAMES
                and self.logged_activity.get(display_id) != "Exited"
            ):
                exit_frame = max(1, frame_number - config.LOST_FRAMES)
                events.append((exit_frame, display_id, "Exited"))
                self.logged_activity[display_id] = "Exited"
                self.pending_activity[display_id] = "Exited"
                self.seen_display_ids.discard(display_id)
                self.previous_positions.pop(display_id, None)

        return events

    def finalize_tracks(self, frame_number):
        events = []

        for display_id in list(self.seen_display_ids):
            if self.logged_activity.get(display_id) != "Exited":
                events.append((frame_number, display_id, "Exited"))
                self.logged_activity[display_id] = "Exited"

        self.seen_display_ids.clear()
        self.missing_frames.clear()
        return events

    def _zone_dwell(self, display_id, zone_name):
        counts = self.zone_dwell.setdefault(display_id, {})
        counts[zone_name] = counts.get(zone_name, 0) + 1
        return counts[zone_name]

    def _reset_other_zone_dwell(self, display_id, active_zone):
        counts = self.zone_dwell.setdefault(display_id, {})
        for zone_name in list(counts.keys()):
            if zone_name != active_zone:
                counts[zone_name] = 0

    def _apply_zone_logic(self, display_id, foot_x, foot_y, norm_move, candidate, is_picking):
        zone_checks = [
            ("checkout", self.zones["checkout"], "Checkout"),
            ("queue", self.zones["queue"], "Queueing"),
            ("shelf", self.zones["shelf"], "Shelf Interaction"),
        ]

        for zone_name, zone, zone_activity in zone_checks:
            if inside_zone(foot_x, foot_y, zone):
                dwell = self._zone_dwell(display_id, zone_name)
                self._reset_other_zone_dwell(display_id, zone_name)

                if zone_name == "shelf":
                    self.shelf_dwell[display_id] = (
                        self.shelf_dwell.get(display_id, 0) + 1
                    )
                    self.was_in_shelf[display_id] = True

                    if is_picking:
                        return "Picking Product"

                    if (
                        norm_move <= config.SLOW_WALK_RATIO
                        and self.shelf_dwell[display_id] >= config.PICKING_FRAMES
                    ):
                        return "Picking Product"
                    if dwell >= config.ZONE_DWELL_FRAMES:
                        return "Shelf Interaction"
                    return candidate

                if zone_name in {"checkout", "queue"} and dwell >= config.ZONE_DWELL_FRAMES:
                    return zone_activity

                return candidate

        if self.was_in_shelf.get(display_id):
            self.was_in_shelf[display_id] = False
            self.shelf_dwell[display_id] = 0
            if norm_move <= config.SLOW_WALK_RATIO:
                return "Returning Product"

        self._reset_other_zone_dwell(display_id, None)

        if (
            candidate == "Standing"
            and self.stationary_frames.get(display_id, 0) >= config.IDLE_FRAMES
        ):
            return "Idle"
        if (
            candidate in {"Standing", "Slow Walking"}
            and self.stationary_frames.get(display_id, 0) >= config.WAITING_FRAMES
        ):
            return "Waiting"

        return candidate

    def _movement_activity(self, norm_move):
        if norm_move > config.RUN_RATIO:
            return "Running"
        if norm_move > config.WALK_RATIO:
            return "Walking"
        if norm_move > config.SLOW_WALK_RATIO:
            return "Slow Walking"
        return "Standing"

    def _stabilize(self, display_id, candidate):
        if candidate in {"Entered", "Exited", "Returning Product"}:
            if self.logged_activity.get(display_id) == candidate:
                return None
            self.pending_activity[display_id] = candidate
            self.candidate_history.setdefault(display_id, []).clear()
            return candidate

        history = self.candidate_history.setdefault(display_id, [])
        history.append(candidate)
        if len(history) > config.STABLE_FRAMES:
            history.pop(0)

        if len(history) == config.STABLE_FRAMES:
            counts = {}
            for c in history:
                counts[c] = counts.get(c, 0) + 1
            most_common = max(counts, key=counts.get)
            
            if counts[most_common] >= config.STABLE_FRAMES // 2:
                self.pending_activity[display_id] = most_common
                return most_common

        return None


class VideoActivityAnalyzer:
    """End-to-end CCTV analysis pipeline."""

    OUTPUT_DIR = config.OUTPUT_FOLDER

    def __init__(self, video_path):
        self.video_path = video_path
        self.width, self.height, self.fps, self.frame_count = get_video_properties(
            video_path
        )
        self.zones = self._build_zones()
        self.recognizer = ActivityRecognizer(self.zones, self.fps)
        self.identity_manager = PersonIdentityManager(
            config.REID_DISTANCE_RATIO,
            config.LOST_FRAMES,
        )
        self.track_validator = TrackValidator(
            config.MIN_CONFIRM_FRAMES,
            config.MIN_TRACK_LIFETIME,
            config.LOST_FRAMES,
        )
        self.timestamp_extractor = CCTVTimestampExtractor(fps=self.fps)
        self.pose_estimator = PoseEstimator()
        self.output_video_path = self._build_output_path()
        self.track_positions = {}  # display_id -> (foot_x, foot_y, box_h)

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

        model_path = ensure_model(config.MODEL_PATH, config.MODEL_FALLBACK)
        model = YOLO(model_path)

        timeline = TimelineGenerator(fps=self.fps, output_path=config.TIMELINE_FILE)
        latest_activity = {}
        writer = None
        capture = None
        frames_written = 0

        try:
            capture = cv2.VideoCapture(self.video_path)
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

            frame_number = 0
            current_timestamp = "00:00:00"
            
            # Profiling accumulators
            time_read = 0.0
            time_ocr = 0.0
            time_infer = 0.0
            time_track = 0.0
            time_write = 0.0
            total_frames = int(self.frame_count) if self.frame_count > 0 else 1

            while True:
                t0 = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    break
                t1 = time.perf_counter()
                time_read += (t1 - t0)

                frame_number += 1
                
                # Extract actual timestamp from video frame
                current_timestamp = self.timestamp_extractor.get_timestamp(frame, frame_number)
                t2 = time.perf_counter()
                time_ocr += (t2 - t1)
                
                # Bypassing Pose Extraction since it's incompatible and saves CPU
                landmarks = None
                
                active_display_ids = set()
                active_raw_ids = set()

                results = model.track(
                    frame,
                    persist=True,
                    tracker=config.TRACKER,
                    conf=config.CONFIDENCE,
                    iou=config.IOU,
                    imgsz=config.INFERENCE_SIZE,
                    classes=[config.PERSON_CLASS],
                    verbose=False,
                )
                t3 = time.perf_counter()
                time_infer += (t3 - t2)

                result = results[0]
                annotated = frame.copy()

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = (
                        result.boxes.id.cpu().numpy().astype(int)
                        if result.boxes.id is not None
                        else np.array([], dtype=int)
                    )

                    filtered_boxes, filtered_conf = filter_detections(
                        boxes,
                        confidences,
                        self.width,
                        self.height,
                        config,
                    )

                    for box, conf in zip(filtered_boxes, filtered_conf):
                        best_track_id = self._match_track_id(box, boxes, track_ids)
                        if best_track_id is None:
                            continue

                        raw_id = int(best_track_id)
                        active_raw_ids.add(raw_id)
                        self.track_validator.update(raw_id, present=True)

                        x1, y1, x2, y2 = map(int, box)
                        foot_x, foot_y = box_foot_point(box)
                        b_height = box_height(box)

                        confirmed = self.track_validator.is_confirmed(raw_id)
                        display_id = self.identity_manager.assign(
                            raw_id,
                            foot_x,
                            foot_y,
                            b_height,
                            frame_number,
                            confirmed,
                        )

                        if display_id is None:
                            continue

                        active_display_ids.add(display_id)

                        prev = self.track_positions.get(display_id)
                        if prev is None:
                            movement_distance = 0.0
                        else:
                            frames_elapsed = max(1, frame_number - prev[3])
                            total_dist = calculate_distance(
                                prev[0],
                                prev[1],
                                foot_x,
                                foot_y,
                            )
                            movement_distance = total_dist / frames_elapsed

                        self.track_positions[display_id] = (foot_x, foot_y, b_height, frame_number)

                        # Determine if picking product via pose
                        is_picking = self.pose_estimator.is_picking_product(landmarks)

                        activity = self.recognizer.recognize(
                            frame_number,
                            display_id,
                            foot_x,
                            foot_y,
                            movement_distance,
                            b_height,
                            is_picking,
                        )

                        if activity:
                            timeline.add_event(current_timestamp, display_id, activity)
                            latest_activity[display_id] = activity
                            self.recognizer.register_logged_activity(
                                display_id,
                                activity,
                            )

                        label = latest_activity.get(display_id, "Tracking")
                        self._draw_annotation(
                            annotated,
                            x1,
                            y1,
                            x2,
                            y2,
                            display_id,
                            label,
                        )

                known_raw_ids = set(self.track_validator.frame_counts.keys())
                for raw_id in known_raw_ids - active_raw_ids:
                    self.track_validator.update(raw_id, present=False)

                for display_id, position in list(self.track_positions.items()):
                    if display_id not in active_display_ids:
                        self.identity_manager.mark_missing(
                            display_id,
                            position[0],
                            position[1],
                            position[2],
                            frame_number,
                        )

                for exit_frame, display_id, activity in (
                    self.recognizer.update_missing_tracks(
                        active_display_ids,
                        frame_number,
                    )
                ):
                    timeline.add_event(current_timestamp, display_id, activity)
                    latest_activity.pop(display_id, None)
                    self.track_positions.pop(display_id, None)
                    self.identity_manager.purge_raw_id(display_id)

                t4 = time.perf_counter()
                time_track += (t4 - t3)

                writer.write(annotated)
                t5 = time.perf_counter()
                time_write += (t5 - t4)
                
                frames_written += 1
                
                if frame_number % 50 == 0:
                    pct = (frame_number / total_frames) * 100
                    avg_read = (time_read / frame_number) * 1000
                    avg_ocr = (time_ocr / frame_number) * 1000
                    avg_infer = (time_infer / frame_number) * 1000
                    avg_track = (time_track / frame_number) * 1000
                    avg_write = (time_write / frame_number) * 1000
                    print(f"Progress: {pct:.1f}% | Latency (ms): Read={avg_read:.1f}, OCR={avg_ocr:.1f}, Infer={avg_infer:.1f}, Track={avg_track:.1f}, Write={avg_write:.1f}", flush=True)

            for exit_frame, display_id, activity in self.recognizer.finalize_tracks(
                frame_number
            ):
                # Pass a generic end timestamp or the last known timestamp
                timeline.add_event(current_timestamp, display_id, activity)

        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            timeline.close()

        if frames_written == 0:
            raise ValueError("No frames were processed from the input video.")

        print(
            f"Confirmed persons: {self.identity_manager.confirmed_person_count}",
            flush=True,
        )
        return self.output_video_path

    @staticmethod
    def _match_track_id(target_box, all_boxes, track_ids):
        if len(track_ids) == 0:
            return None

        best_iou = 0.0
        best_id = None

        for box, track_id in zip(all_boxes, track_ids):
            x1 = max(target_box[0], box[0])
            y1 = max(target_box[1], box[1])
            x2 = min(target_box[2], box[2])
            y2 = min(target_box[3], box[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, target_box[2] - target_box[0]) * max(
                0.0, target_box[3] - target_box[1]
            )
            area_b = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        if best_iou >= 0.5:
            return best_id

        return None

    @staticmethod
    def _draw_annotation(frame, x1, y1, x2, y2, display_id, activity):
        color = (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{person_label(display_id)} | {activity}"
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
