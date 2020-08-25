import numpy as np
import math
import functools as fu
import cv2
import random as rand


def transform_points(m, points):
    """ It transforms the given point/points using the given transformation matrix.
    :param points: numpy array, list
            The point/points to be transformed given the transformation matrix.

    :param m: An 3x3 matrix
            The transformation matrix which will be used for the transformation.
    :return: The transformed point/points.
    """
    ph = make_homogeneous(points).T
    ph = m @ ph
    return make_euclidean(ph.T)


def transform_image(image, m):
    """ It transforms the given image using the given transformation matrix.
    :param img: An image
            The image to be transformed given the transformation matrix.
    :param m: An 3x3 matrix
            The transformation matrix which will be used for the transformation.
    :return: The transformed image.
    """
    row, col, _ = image.shape
    return cv2.warpPerspective(image, m, (col, row))


def make_homogeneous(points):
    """ It converts the given point/points in an euclidean coordinates into a homogeneous coordinate
        :param points: numpy array, list
                The point/points to be converted into a homogeneous coordinate.
        :return: The converted point/points in the homogeneous coordinates.
    """
    if isinstance(points, list):
        points = np.asarray([points], dtype=np.float64)
        return np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    else:
        return np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))


def make_euclidean(points):
    """It converts the given point/points in a homogeneous coordinate into an euclidean coordinates.
    :param points: numpy array, list
            The point/points to be converted into an euclidean coordinates.
    :return: The converted point/points in the euclidean coordinates.
    """
    return points[:, :-1]


def identity():
    """ It provides an identity transformation matrix.
    :return: An identity matrix (3 x 3) using homogeneous coordinates.
    """
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]],
                    dtype=np.float64)

def rotating(θ=0):
    """ It provides a rotation matrix given θ degrees which can then be used to rotate 2D point/points or an image
    clockwise about the origin. If you want to rotate counterclockwise pass a negative degree.
    :param θ: int
        The amount of degree to be rotated. The default value is 0 which means when using it to rotate it won't rotate
        the point/points or the image at all.
    :returns: The rotation matrix (3 x 3) using homogeneous coordinates.
    """
    θ = np.radians(θ)
    cos = math.cos(θ)
    sin = math.sin(θ)
    return np.array([[cos, sin, 0],
                     [-sin, cos, 0],
                     [0, 0, 1]],
                    dtype=np.float64)


def translating(t_x=0, t_y=0):
    """ It provides a translate matrix given quantity t_x and t_y for shifting x and y axes respectively.It can then
    be used to translate or ”shift” a 2D point/points or an image.
    as well as the y-axis by t_y.
    :param t_x: int
        The amount of shifting in the direction of the x-axis
    :param t_y: int
        The amount of shifting in the direction of the y-axis
    The default values for both are 0. That is it does not translate the point/points or the image when applied.
    :returns: The translation matrix (3 x 3) in homogeneous coordinates.
    """
    return np.array([[1, 0, t_x],
                     [0, 1, t_y],
                     [0, 0, 1]],
                    dtype=np.float64)


def scaling(scale_x=1, scale_y=1):
    """ It provides a scale matrix given quantity scale_x and scale_y for scaling x and y axes respectively.It can then
    be used to scale a 2D point/points or an image.

    scales (enlarge or shrink) the given 2D point/points in the direction of the x-axis by scale_x
    as well as the y-axis by scale_x.
    :param scale_x: int
        The scale factor by which we wish to enlarge/shrink the point/points in the direction of the x-axis.
    :param scale_y: int
        The scale factor by which we wish to enlarge/shrink the point/points in the direction of the y-axis.
    The default values for both are 1. That is it does not scale the point/points or the image when applied.
    :return: The scaling matrix (3 x 3) in homogeneous coordinates.
    """
    return np.array([[scale_x, 0, 0],
                     [0, scale_y, 0],
                     [0, 0, 1]],
                    dtype=np.float64)


def arbitrary():
    """
    :return: An (3 x 3) arbitrary transformation matrix using translating, scaling and rotating function randomly.
    """
    θ = rand.randint(-360, 361)
    r = rotating(θ)
    sx, sy = rand.sample(range(-10, 11), 2)
    s = scaling(sx, sy)
    tx, ty = rand.sample(range(-400, 401), 2)
    t = translating(tx, ty)
    I = identity()
    if 0 <= tx <= 200:
        return s @ t @ r @ I
    else:
        return r @ s @ I @ t


def invert(m):
    """ It provides a matrix for performing the inversion.
    :param m: a (3 x 3) matrix.
    :return: The inverse of the given matrix.
    """
    d = np.linalg.det(m)
    if d != 0:
        return np.linalg.inv(m).astype(dtype=np.float64)
    else:
        raise Exception("It is a non-invertible matrix")


def combine(*transformations):
    """ It combines the given transformation matrices.
    Be aware of which order you are passing the transformation matrices since it will be used to transform in that order.
    :param transformations: (3 x 3) transformation matrices. As many as you want.
        The matrices to be combined.
    :return: The combined matrix (3 x 3).
    """
    transformations = reversed(transformations)
    return fu.reduce(lambda tr1, tr2: tr1 @ tr2, transformations)


def learn_affine(srs, tar):
    """ It finds the affine transformation matrix between the two given triangles (3 points).
    A x = b     =>   x = inv(A) b
    :param srs: three 2D points in homogeneous coordinates representing a triangle.
        The source points.
    :param tar: three 2D points in homogeneous coordinates representing a triangle.
        The target pints.
    :return: The affine transformation matrix.
    """
    x1, x2, x3 = srs[0, 0], srs[1, 0], srs[2, 0]
    y1, y2, y3 = srs[0, 1], srs[1, 1], srs[2, 1]
    b = tar.flatten()
    a = np.array([[x1, y1, 1, 0, 0, 0],
                  [0, 0, 0, x1, y1, 1],
                  [x2, y2, 1, 0, 0, 0],
                  [0, 0, 0, x2, y2, 1],
                  [x3, y3, 1, 0, 0, 0],
                  [0, 0, 0, x3, y3, 1]],
                 dtype=np.float64)

    d = np.linalg.det(a)
    if d != 0:
        ai = np.linalg.inv(a)
        x = ai @ b
        x = x.flatten()
        a1, a2, a3, a4 = x[0], x[1], x[3], x[4]
        tx, ty = x[2], x[5]
        aff_transformation = np.array([[a1, a2, tx],
                                       [a3, a4, ty],
                                       [0, 0, 1]],
                                      dtype=np.float64)
        return aff_transformation
    else:
        raise Exception("It is a non-invertible matrix")