import os
import json

import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def load_json(directory, filename):
    """Load json file from subdirectory in "inputs/images" with the given filename
    - without .json extension!

    Returns: The json data as a dictionary or array (depending on the file).
    """
    with open(os.path.join('inputs/images', directory, f'{filename}.json')) as file:
        return json.load(file)


def load_images(directory):
    """Load images from a subdirectory in "inputs/images" using OpenCV.

    Returns: The list of loaded images in order.
    """
    with open(os.path.join('inputs/images', directory, 'positions.json')) as file:
        screen_points = json.load(file)

    images = [cv2.imread(os.path.join(
        'inputs/images', directory, f'{i}.jpg')) for i in range(len(screen_points))]

    return images


def dist(a, b):
    """Calculate the euclidean distance from a to b.
    """
    return np.linalg.norm(a-b)


def pupil_json_to_opencv(pupil):
    """Convert pupil loaded from json file to the format used by OpenCV,
    i.e. ((cx, cy), (ax, ay), angle)

    Returns: Tuple containing ((cx, cy), (ax, ay), angle)
    """
    p = pupil
    return ((p['cx'], p['cy']), (p['ax'], p['ay']), p['angle'])


def pupil_to_int(pupil):
    """Convert pupil parameters to integers. Useful when drawing.
    """
    p = pupil
    return ((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), int(p[2]))
