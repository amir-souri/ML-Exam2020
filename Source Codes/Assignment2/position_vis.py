import os
import sys
import cv2
import numpy as np
import gaze
from utils import load_json, load_images
from record import PictureRecorder

def main():
    """Visualise pupil/glint data and detected estimates.
    * Blue shapes represent annotated data.
    * Red shapes represent detected estimates.
    """
    if len(sys.argv) == 1:
        raise ValueError("You must supply data folder as a program argument.")

    PositionVisualiser(sys.argv[1]).loop()


class PositionVisualiser:

    def __init__(self, dataset):
        self.dataset = os.path.join('inputs', 'images', dataset)
        self.pos = np.array(load_json(dataset, 'positions'))
        self.title = 'Preview'
        cv2.namedWindow(self.title)
        cv2.createTrackbar('Image', self.title, 0,
                           len(self.pos), self.change_idx)

        # self.model = gaze.GazeModel(load_images(dataset)[:9], self.pos[:9])
        self.model = gaze.PolynomialGaze(load_images(dataset)[:9], self.pos[:9], order=2 )

        self.update(0)

    def change_idx(self, value):
        self.update(value)

    def loop(self):
        while True:
            k = cv2.waitKey(0)

            if k == ord('s'):
                PictureRecorder('outputs', self.img, 'gaze')
                PictureRecorder('outputs', self.screen, 'screen')
            elif k == ord('q'):
                return

    def update(self, idx):
        img = cv2.imread(os.path.join(self.dataset, f'{idx}.jpg'))
        if img is None:
            return

        #c, ax, angle = detector.find_pupil(img)
        #gs = detector.find_glints(img, c)
        y, x = self.pos[idx]

        screen = np.zeros((1080//4, 1920//4, 3), dtype=np.uint8)

        for i in range(9, len(self.pos)):
            if i == idx:
                continue

            cv2.drawMarker(
                screen, (self.pos[i][1]//4, self.pos[i][0]//4), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

        for i in range(9):
            if i == idx:
                continue

            cv2.drawMarker(
                screen, (self.pos[i][1]//4, self.pos[i][0]//4), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

        cv2.drawMarker(screen, (x//4, y//4), (0, 0, 255),
                       cv2.MARKER_CROSS, 15, 2)

        x, y = self.model.estimate(img)
        cv2.drawMarker(screen, (int(x)//4, int(y)//4),
                       (0, 255, 255), cv2.MARKER_CROSS, 25, 1)

        cv2.imshow(self.title, img)
        cv2.imshow('Screen', screen)

        self.img = img
        self.screen = screen


if __name__ == '__main__':
    main()
