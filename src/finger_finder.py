import cv2
import math
import argparse
import numpy as np
from urllib.request import urlopen


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path of image")

    parser.add_argument('-url',
                        action='store_true')

    parser.add_argument('-path',
                        action='store_true')

    return parser


def url_to_img(url):
    req = urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)

    return image


def get_mid_point(p1, p2):
    mid_pnt = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    mid_x = mid_pnt[0]
    mid_y = mid_pnt[1]

    return mid_x, mid_y


def get_max_y_point(list_of_points):
    max_y_point = max(list_of_points, key=lambda x: x[1])

    return max_y_point


def sorted_points(list_of_points):
    sorted_points_list = sorted(list_of_points, key=lambda x: x[0])

    return sorted_points_list


def finger_finder(img_path):

    if url_flag:
        frame = url_to_img(img_path)
    else:
        frame = cv2.imread(img_path)

    lower = np.array([0, 40, 73], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=lambda contour: cv2.contourArea(contour))

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    red_point_coordinates = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

        d = (2 * ar) / a

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90 and d > 30:
            red_point_coordinates.append(far)
            # cv2.line(frame, start, end, [0, 255, 0], 2)
            cv2.circle(frame, far, 5, [0, 0, 255], -1)

    cnt_box = cv2.boundingRect(cnt)
    x, y, w, h = cnt_box

    m_x, m_y = get_mid_point((x, y), (x + w, y + h))

    max_x, max_y = get_max_y_point(red_point_coordinates)
    point_list = sorted_points(red_point_coordinates)

    if max_x > m_x:

        points = [point_list[0], point_list[1]]
        dist = math.sqrt((point_list[1][0] - point_list[0][0]) ** 2 + (point_list[1][1] - point_list[0][1]) ** 2)

    else:
        points = [point_list[2], point_list[3]]
        dist = math.sqrt((point_list[3][0] - point_list[2][0]) ** 2 + (point_list[3][1] - point_list[2][1]) ** 2)

    return points, dist


if __name__ == "__main__":
    p = get_parser().parse_args()

    img_path = p.img_path
    url_flag = p.url
    path_flag = p.path

    detected_points, distance = finger_finder(img_path)

    if url_flag:
        img = url_to_img(img_path)
    elif path_flag:
        img = cv2.imread(img_path)
    cv2.circle(img, detected_points[0], 5, [0, 0, 255], -1)
    cv2.circle(img, detected_points[1], 5, [0, 0, 255], -1)
    cv2.line(img, detected_points[0], detected_points[1], [0, 255, 0], 2)
    cv2.putText(img, f'Distance = {distance}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'Coordinates = {detected_points}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite('result.png', img)
