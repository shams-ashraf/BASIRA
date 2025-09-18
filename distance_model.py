import math

def estimate_distance_object(obj_box, label, frame_width=320, focal_length=500):
    known_widths = {
        "person": 0.5,
        "traffic light": 0.3,
        "car": 1.8,
        "bicycle": 0.6
    }

    x1, y1, x2, y2 = obj_box
    obj_pixel_width = max(1, x2 - x1)

    if label in known_widths:
        real_width = known_widths[label]
    else:
        real_width = 0.5

    distance = (real_width * focal_length) / obj_pixel_width

    if distance < 2:
        return f"{distance:.1f} m - CLOSE"
    elif distance < 5:
        return f"{distance:.1f} m - MEDIUM"
    else:
        return f"{distance:.1f} m - FAR"
