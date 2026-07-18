# Video
VIDEO_PATH = "videos/test.mp4"

# Model – YOLO11 with YOLOv8 fallback handled in ensure_model()
MODEL_PATH = "models/yolo11n.pt"
MODEL_FALLBACK = "models/yolov8n.pt"

# Output
OUTPUT_FOLDER = "outputs"
TIMELINE_FILE = "data/timeline.csv"

# Detection
PERSON_CLASS = 0
CONFIDENCE = 0.52
IOU = 0.5
INFERENCE_SIZE = 1280

# Tracking
TRACKER = "models/bytetrack_chrono.yaml"

# Detection filtering (reduce false positives)
MIN_BOX_AREA_RATIO = 0.0015
MAX_BOX_AREA_RATIO = 0.20
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 5.5
DUPLICATE_IOU = 0.65

# Track validation
MIN_CONFIRM_FRAMES = 18
MIN_TRACK_LIFETIME = 25
LOST_FRAMES = 150
REID_DISTANCE_RATIO = 4.0

# Activity stabilization
STABLE_FRAMES = 12
ZONE_DWELL_FRAMES = 15
PICKING_FRAMES = 30
WAITING_FRAMES = 60
IDLE_FRAMES = 120

# Movement thresholds (fraction of bbox height per frame)
RUN_RATIO = 0.06
WALK_RATIO = 0.03
SLOW_WALK_RATIO = 0.015
