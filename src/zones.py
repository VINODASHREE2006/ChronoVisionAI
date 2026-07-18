# Reference resolution used when defining retail zones.
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160

# Entrance / exit door area (bottom-right of frame).
ENTRANCE_ZONE = (3000, 1200, 3840, 2160)

# Main shelf browsing area.
SHELF_ZONE = (0, 0, 3000, 1700)

# Checkout counter area (center-right).
CHECKOUT_ZONE = (2400, 1400, 3300, 2100)

# Queue/waiting lane in front of checkout.
QUEUE_ZONE = (2100, 1500, 2500, 2100)

# Exit zone overlaps the entrance door.
EXIT_ZONE = (3000, 1200, 3840, 2160)
