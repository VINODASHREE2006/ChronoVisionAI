def frame_to_time(frame, fps):
    """
    Convert frame number to HH:MM:SS format.
    """

    seconds = int(frame / fps)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


def calculate_distance(x1, y1, x2, y2):
    """
    Euclidean distance between two points.
    """

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def inside_zone(x, y, zone):
    """
    Check whether a point lies inside a rectangular zone.
    """

    x1, y1, x2, y2 = zone

    return (
        x1 <= x <= x2 and
        y1 <= y <= y2
    )