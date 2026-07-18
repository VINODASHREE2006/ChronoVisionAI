import os

import cv2
import numpy as np


def frame_to_time(frame, fps):
    """Convert frame number to HH:MM:SS format."""

    if fps <= 0:
        fps = 30

    seconds = int(frame / fps)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


def calculate_distance(x1, y1, x2, y2):
    """Euclidean distance between two points."""

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def box_iou(box_a, box_b):
    """Intersection-over-union for xyxy boxes."""

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def box_center(box):
    """Return center point of an xyxy box."""

    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_foot_point(box):
    """Return bottom-center point used for zone checks."""

    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def box_height(box):
    return max(1.0, box[3] - box[1])


def inside_zone(x, y, zone):
    """Check whether a point lies inside a rectangular zone."""

    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def scale_zone(zone, src_width, src_height, dst_width, dst_height):
    """Scale a zone defined for one resolution to another."""

    if src_width <= 0 or src_height <= 0:
        return zone

    x1, y1, x2, y2 = zone
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height

    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def get_video_properties(video_path):
    """Return width, height, fps, and frame count for a video file."""

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        capture.release()
        return 0, 0, 30, 0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0 or fps != fps:
        fps = 30

    return width, height, fps, frame_count


def ensure_model(model_path="models/yolo11n.pt", fallback="models/yolov8n.pt"):
    """Return the best available local/downloadable model path."""

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if os.path.exists(model_path):
        return model_path

    if os.path.exists(fallback):
        return fallback

    for candidate in ("yolo11n.pt", "yolov8n.pt"):
        try:
            from ultralytics import YOLO

            YOLO(candidate)
            return candidate
        except Exception:
            continue

    return "yolov8n.pt"


def person_label(display_id):
    """Format a stable display id as a readable person label."""

    return f"Person {int(display_id)}"


def normalize_timeline_columns(df):
    """Normalize timeline CSV column names across legacy formats."""

    rename_map = {}

    for column in df.columns:
        normalized = column.strip().lower().replace("_", " ")
        if normalized in {"time", "timestamp"}:
            rename_map[column] = "Timestamp"
        elif normalized in {"person id", "person", "person_id"}:
            rename_map[column] = "Person"
        elif normalized in {"activity", "event"}:
            rename_map[column] = "Activity"

    return df.rename(columns=rename_map)


def filter_detections(boxes, confidences, frame_width, frame_height, config):
    """Remove low-quality and duplicate person detections."""

    if boxes is None or len(boxes) == 0:
        return np.array([]), np.array([])

    frame_area = max(frame_width * frame_height, 1)
    kept_boxes = []
    kept_conf = []

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area_ratio = (width * height) / frame_area
        aspect_ratio = height / max(width, 1.0)

        if conf < config.CONFIDENCE:
            continue
        if area_ratio < config.MIN_BOX_AREA_RATIO:
            continue
        if area_ratio > config.MAX_BOX_AREA_RATIO:
            continue
        if aspect_ratio < config.MIN_ASPECT_RATIO:
            continue
        if aspect_ratio > config.MAX_ASPECT_RATIO:
            continue

        kept_boxes.append(box)
        kept_conf.append(conf)

    if not kept_boxes:
        return np.array([]), np.array([])

    kept_boxes = np.array(kept_boxes)
    kept_conf = np.array(kept_conf)

    order = np.argsort(-kept_conf)
    kept_boxes = kept_boxes[order]
    kept_conf = kept_conf[order]

    final_boxes = []
    final_conf = []

    for box, conf in zip(kept_boxes, kept_conf):
        duplicate = False
        for kept in final_boxes:
            if box_iou(box, kept) >= config.DUPLICATE_IOU:
                duplicate = True
                break
        if not duplicate:
            final_boxes.append(box)
            final_conf.append(conf)

    if not final_boxes:
        return np.array([]), np.array([])

    return np.array(final_boxes), np.array(final_conf)
