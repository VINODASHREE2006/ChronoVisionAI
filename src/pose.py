# Fallback heuristics for action detection since MediaPipe solutions is deprecated/missing on this env.

class PoseEstimator:
    def __init__(self):
        pass

    def process_frame(self, frame):
        return None

    def extract_landmarks(self, results, frame_width, frame_height):
        return None

    def is_picking_product(self, landmarks=None, box=None):
        """
        Fallback heuristic without MediaPipe:
        If we want to detect picking product, we can use the aspect ratio of the bounding box
        or simply return False and let the zone dwell logic handle it.
        """
        # For now, we rely on the improved ZONE DWELL logic in ActivityRecognizer
        return False

    def is_bending(self, landmarks=None):
        return False
