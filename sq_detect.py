
import cv2
import numpy as np
import math
from operator import itemgetter

# ================= COLOR RANGES (HSV) =================
color_lims = {
    'G': ((50, 100, 100), (90, 255, 240)),
    'R': ((170, 110, 110), (10, 255, 155)),
    'W': ((0, 0, 120), (255, 50, 255)),
    'Y': ((11, 80, 150), (40, 255, 255)),
    'B': ((95, 60, 70), (120, 255, 220)),
    'O': ((175, 180, 155), (10, 255, 255))
}

# ================= DETECTION SETTINGS =================
DS_SQUARE_SIDE_RATIO = 1.5
DS_MORPH_KERNEL_SIZE = 5
DS_MIN_AREA_RATIO = 0.7
DS_MIN_SQUARE_SIZE = 0.08
DS_MAX_SQUARE_SIZE = 0.30


# ======================================================
def index_to_cube(pts):
    if len(pts) != 9:
        return None

    pts = [list(pts[i]) + [i] for i in range(len(pts))]
    pts.sort(key=itemgetter(1))

    mat = [[pts[3 * i + j] for j in range(3)] for i in range(3)]

    for i in range(3):
        mat[i].sort(key=itemgetter(0))

    for i in range(3):
        for j in range(3):
            mat[i][j] = mat[i][j][2]

    return mat


# ======================================================
def get_color_mask(im, color_name):

    color_lim = color_lims[color_name]

    # Handle red/orange wrap-around
    if color_name in ['R', 'O']:
        big_hue = max(color_lim[0][0], color_lim[1][0])
        small_hue = min(color_lim[0][0], color_lim[1][0])

        lower1 = np.array([0, color_lim[0][1], color_lim[0][2]])
        upper1 = np.array([small_hue, color_lim[1][1], color_lim[1][2]])

        lower2 = np.array([big_hue, color_lim[0][1], color_lim[0][2]])
        upper2 = np.array([180, color_lim[1][1], color_lim[1][2]])

        mask = cv2.inRange(im, lower1, upper1) | cv2.inRange(im, lower2, upper2)

    else:
        lower = np.array(color_lim[0])
        upper = np.array(color_lim[1])
        mask = cv2.inRange(im, lower, upper)

    # Morphological cleaning
    kernel = np.ones((DS_MORPH_KERNEL_SIZE, DS_MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ======================================================
def detect_square(im, color_name):

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, color_name)

    # ✅ FIXED for OpenCV 4
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None:
        return []

    valid_contours = []

    for cont in contours:
        rect = cv2.minAreaRect(cont)
        length, breadth = rect[1]

        if length == 0 or breadth == 0:
            continue

        ratio = max(length / breadth, breadth / length)

        if ratio > DS_SQUARE_SIDE_RATIO:
            continue

        area_ratio = cv2.contourArea(cont) / (length * breadth)

        if area_ratio < DS_MIN_AREA_RATIO:
            continue

        size = max(length, breadth)

        if not (DS_MIN_SQUARE_SIZE * im.shape[0] < size < DS_MAX_SQUARE_SIZE * im.shape[0]):
            continue

        valid_contours.append(rect)

    return valid_contours


# ======================================================
def get_cube_state(im):

    colors_detected = []
    cube_state = [[None] * 3 for _ in range(3)]

    for color in color_lims:
        rects = detect_square(im, color)

        for rect in rects:
            colors_detected.append((color, rect))

    if len(colors_detected) != 9:
        return None

    index_mat = index_to_cube([prop[1][0] for prop in colors_detected])

    if index_mat is None:
        return None

    for i in range(3):
        for j in range(3):
            cube_state[i][j] = colors_detected[index_mat[i][j]][0]

    return cube_state


# ======================================================
def draw_rects(im, rects, colour=(0, 255, 0)):

    im2 = im.copy()

    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.polylines(im2, [box], True, colour, 2)

    return im2
