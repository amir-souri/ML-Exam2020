import cv2
import numpy as np
from utils import dist

def find_pupil(img, debug=True):
    """Detects and returns a single pupil candidate for a given image.
    You can use the debug flag for showing threshold images, print messages, etc.

    Returns: A pupil candidate in OpenCV ellipse format.
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_grey, 50, 255, 1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(c)
        if (len(approx) > 7) & (len(approx) < 21) & (area > 50) & (area < 800):
            contour_list.append(c)

    if len(contour_list) is not 0:
        areas = [cv2.contourArea(c) for c in contour_list]
        max_index = np.argmax(areas)
        max_contour = contour_list[np.asscalar(max_index)]
        ellipse = cv2.fitEllipse(max_contour)
    else:
        raise Exception("Does not detect any contour!")

    if debug == True:
        cv2.imshow("find_pupil thresholded image", thresh)
        cv2.waitKey(100)

    return ellipse


def find_glints(img, center, debug=True):
    """Detects and returns up to four glint candidates for a given image.
    You can use the debug flag for showing threshold images, print messages, etc.

    Returns: Detected glint positions.
    """
    center_coordinates = (int(center[0]), int(center[1]))
    empty = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_iris = cv2.circle(empty, center=center_coordinates, radius=37, color=(255, 255, 255), thickness=-1)  # 37
    img_mask_iris = cv2.bitwise_and(img, img, mask=mask_iris)
    img_grey = cv2.cvtColor(img_mask_iris, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_grey, 160, 255, 0)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) is not 0:
        my_glints = np.zeros((len(contours), 2))
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            my_glints[i] = [cX, cY]
    else:
        raise Exception("Does not detect any contour!")

    if len(my_glints) is not 0:
        dis = np.zeros(len(my_glints))
        for i in range(len(my_glints)):
            dis[i] = dist(np.asarray(my_glints[i]), np.asarray(center_coordinates))

        indexes = np.zeros(4, dtype=int)
        for i in range(4):
            indexes[i] = np.argmin(dis)
            dis[indexes[i]] = np.inf

        res = np.zeros((4, 2))
        for i in range(4):
            res[i] = my_glints[indexes[i]]

        rem_res = remove_duplicates(res)

    else:
        raise Exception('Does not detect any glint!')

    if debug == True:
        cv2.imshow("find_glints thresholded image", thresh)
        cv2.imshow("find_glints erosion-dilation (opening)", dilation)
        cv2.imshow("find_glints mask_iris", img_mask_iris)
        cv2.waitKey(1000)
        print('my_glints', my_glints)
        print('center_coordinates', center_coordinates)
        print("glint candidates", rem_res)

    return rem_res


def remove_duplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]