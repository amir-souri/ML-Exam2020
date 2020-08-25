import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import cv2


class SliderPlot:
    """Class for creating an interactive plot for PCA.
    """
    def __init__(self, principal_components, std, mu, inverse_transform):
        """Initialises the plot window and elements.

        Args:
            principal_components: Used when inverse transforming embedded vectors.
            std: Standard deviation of each principal component.
            mu: Mean vector used for inverse transformations.
            inverse_transform: The function used to calculate the inverse transformation.
        """
        self.principal_components = principal_components
        self.n_components = principal_components.shape[1]
        self.mu = mu
        self.inverse_transform = inverse_transform
        self.vector = np.zeros((self.n_components))

        fig, ax = plt.subplots()
        self.fig = fig
        plt.subplots_adjust(right=0.7)

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)

        ax.invert_yaxis()

        slider_ax = [
            fig.add_axes([0.72, 0.1 + 0.05 * i, 0.25, 0.03])
            for i in range(self.n_components)
        ]

        self.sliders = [
            Slider(ax, str(i), -std[i] * 3, std[i] * 3, 0)
            for i, ax in enumerate(slider_ax)
        ]
        for s in self.sliders:
            s.on_changed(self.update)

        self.line = ax.scatter([], [])
        self.update(None)

    def update(self, val):
        """Updates the face plot when a slider position has changed.

        Args:
            val: Unused but necessary for the matplotlib Slider api.
        """
        for i in range(self.n_components):
            self.vector[i] = self.sliders[i].val
        t = self.inverse_transform(self.vector, self.principal_components,
                                   self.mu)

        t = t.reshape((-1, 2))
        self.line.set_offsets(t)

        self.fig.canvas.draw_idle()


def image_windows(src, size=(20, 20), stride=(10, 10)):
    """Generate lists of image windows.

    Args:
        src: Image as ndarray.
        size (tuple, optional): Window size (y, x). Defaults to (20, 20).
        stride (tuple, optional): Window strid (y, x). Defaults to (10, 10).

    Returns:
        Returns the list of image patches.
    """
    w = []
    f = []
    height, width = src.shape[:2]
    for y in range(0, height, stride[0]):
        for x in range(0, width, stride[1]):
            window = src[y:y + size[0], x:x + size[1]]
            m, n = window.shape[:2]
            if (m == size[0] and n == size[1]):
                w.append(window.flatten())

    return w


def plot_image_windows(vecs, title, size=(10, 10)):
    """Plot window vectors in a grid.

    Args:
        vecs (array-like): List of window vectors.
        title (string): Plot title.
        size (tuple, optional): Plot size. Defaults to (10, 10).
    """
    cols = max(2, int(np.ceil(np.sqrt(len(vecs)))))
    rows = max(2, int(np.ceil((len(vecs))/cols)))

    print(len(vecs))

    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))
    for y in range(rows):
        for x in range(cols):
            ax[y, x].axis('off')

    fig.suptitle(title)

    for i, center in enumerate(vecs):
        img = np.uint8(center.reshape(size[0], size[1], 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i//cols, i%cols].imshow(img)