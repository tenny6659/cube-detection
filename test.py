
import cv2
import numpy as np
from sq_detect import detect_square


# =========================
# 🔧 CAMERA PARAMETERS
# =========================

FOCAL_LENGTH = 700
REAL_CUBE_WIDTH = 6.0   # cm


def estimate_distance(pixel_width):
    if pixel_width == 0:
        return None
    return (REAL_CUBE_WIDTH * FOCAL_LENGTH) / pixel_width


# =========================
# 🎥 START CAMERA
# =========================

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # IMPORTANT: everything must be inside loop
    colors = ["R", "G", "B", "O", "W", "Y"]
    rects = []

    for c in colors:
        r = detect_square(frame, c)
        if r is not None:
            rects.extend(r)

    centers = []
    angles = []
    sizes = []

    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(display, [box], 0, (0, 255, 0), 2)

        (cx, cy), (w, h), angle = rect

        centers.append((cx, cy))
        angles.append(angle)
        sizes.append((w + h) / 2)

    print("Squares detected:", len(centers))

    if len(centers) >= 6:

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        pixel_width = max_x - min_x
        distance = estimate_distance(pixel_width)

        avg_angle = np.mean(angles)
        avg_square_size = np.mean(sizes)

        cv2.rectangle(display,
                      (min_x, min_y),
                      (max_x, max_y),
                      (0, 255, 0), 2)

        cv2.putText(display,
                    f"Squares: {len(centers)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.putText(display,
                    f"Pixel Width: {int(pixel_width)} px",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.putText(display,
                    f"Avg Square Size: {avg_square_size:.1f} px",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.putText(display,
                    f"Angle: {avg_angle:.2f} deg",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        if distance:
            cv2.putText(display,
                        f"Distance: {distance:.2f} cm",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    cv2.imshow("Cube Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
