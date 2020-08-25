import numpy as np
import utils as ut
import detector as det
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class GazeModel:
    """Linear regression model for gaze estimation.
    """

    def __init__(self, calibration_images, calibration_positions):
        """Uses calibration_images and calibratoin_positions to
        create regression mode.
        """
        self.images = calibration_images
        self.positions = calibration_positions
        self.calibrate()

    def calibrate(self):
        """Create the regression model here.
        """

        pups = [det.find_pupil(self.images[i], debug=False) for i in range(len(self.images))]
        pups_centers = np.asarray([pups[i][0] for i in range(len(self.images))])
        pups_centers = np.asarray([[np.round(i) for i in nested] for nested in pups_centers])
        targets_X = self.positions[:, 0]
        targets_Y = self.positions[:, 1]
        D = np.hstack((pups_centers, np.ones((pups_centers.shape[0], 1), dtype=pups_centers.dtype)))
        theta_X, *_ = np.linalg.lstsq(D, targets_X, rcond=None)
        theta_Y, *_ = np.linalg.lstsq(D, targets_Y, rcond=None)
        return theta_X, theta_Y

    def estimate(self, image):
        """Given an input image, return the estimated gaze coordinates.
        """
        my_pup = det.find_pupil(image, debug=False)
        center = np.asarray([np.asarray(my_pup[0])])
        D = np.hstack((center, np.ones((center.shape[0], 1), dtype=center.dtype)))
        t1, t2 = self.calibrate()
        x, y = D @ t1, D @ t2

        return [y[0], x[0]]


class PolynomialGaze():
    """Polynomial regression model for gaze estimation.
        """

    def __init__(self, calibration_images, calibration_positions, order):
        """Uses calibration_images and calibratoin_positions to
        create regression model.
        """
        self.order = order
        self.images = calibration_images
        self.positions = calibration_positions
        self.calibrate()

    def calibrate(self):
        """Create the regression model here.
        """
        my_pups = [det.find_pupil(self.images[i], debug=False) for i in range(len(self.images))]
        my_pups_int = [ut.pupil_to_int(my_pups[i]) for i in range(len(self.images))]
        centers = [my_pups_int[i][0] for i in range(len(self.images))]
        centers_np = np.asarray(centers)
        targets = self.positions
        poly_reg = PolynomialFeatures(degree=self.order)
        centers_np_transformed = poly_reg.fit_transform(centers_np)
        regressor = LinearRegression()
        model_poly = regressor.fit(centers_np_transformed, targets)

        return model_poly, poly_reg

    def estimate(self, image):
        """Given an input image, return the estimated gaze coordinates.
        """
        my_pup = det.find_pupil(image, debug=False)
        my_pup_int = ut.pupil_to_int(my_pup)
        center = my_pup_int[0]
        center_np = np.asarray([center])
        model_poly, poly_reg = self.calibrate()
        pred = model_poly.predict(poly_reg.fit_transform(center_np))

        return np.asscalar(pred[0, 1]), np.asscalar(pred[0, 0])