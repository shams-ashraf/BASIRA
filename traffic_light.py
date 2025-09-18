import cv2

def detect_traffic_lights(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Color ranges
    red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) \
             + cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))

    # Areas
    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    green_area = cv2.countNonZero(green_mask)

    if red_area > yellow_area and red_area > green_area:
        return "RED"
    elif yellow_area > red_area and yellow_area > green_area:
        return "YELLOW"
    elif green_area > red_area and green_area > yellow_area:
        return "GREEN"
    else:
        return "UNKNOWN"
