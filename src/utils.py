import os

import cv2


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
    """Return width, height, and fps for a video file."""

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        capture.release()
        return 0, 0, 30

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if fps <= 0 or fps != fps:
        fps = 30

    return width, height, fps


def ensure_model(model_path="models/yolov8n.pt"):
    """Return a local model path or the default downloadable weights name."""

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if os.path.exists(model_path):
        return model_path

    # Ultralytics downloads yolov8n.pt automatically on first use.
    return "yolov8n.pt"


def person_label(track_id):
    """Format a tracker id as a readable person label."""

    return f"Person {int(track_id)}"


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
