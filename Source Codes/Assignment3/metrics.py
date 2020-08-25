import numpy as np
import torch


class MetricLogger:
    """Enables simple result logging and metric calculation for machine learning models."""

    def __init__(self, one_hot=True):
        """Initialize logger with empty confusion matrix.

        :param one_hot: Specifies whether predictions are one-hot encoded or not.
        """
        self.mat = np.zeros((10, 10))
        self.one_hot = one_hot

    def log(self, predicted, target):
        """Log results for provided arguments. Results are added to the confusion matrix, making it possible to do
        incremental logging.

        :param predicted: Model predictions.
        :param target: Ground-truth labels.
        """
        if type(predicted) is torch.Tensor:
            predicted = predicted.detach().numpy()
        if type(target) is torch.Tensor:
            target = target.detach().numpy()

        if self.one_hot:
            predicted = np.argmax(predicted, axis=1)

        for pi, ti in zip(predicted, target):
            self.mat[pi, ti] += 1

    def reset(self):
        """Reset the logger's internal confusion matrix.
        """
        self.mat = np.zeros(self.mat.shape)

    @property
    def accuracy(self) -> np.ndarray:
        diagonal_sum = self.mat.trace()
        sum_of_all_elements = self.mat.sum()
        return diagonal_sum / sum_of_all_elements

    @property
    def precision(self) -> np.ndarray:
        n = self.mat.shape
        diagonals = np.diag(self.mat)
        rows = np.sum(self.mat, axis=1)
        precision = [diagonals[i] / rows[i] for i in range(n[0])]
        return np.array(precision)

    @property
    def recall(self) -> np.ndarray:
        n = self.mat.shape
        diagonals = np.diag(self.mat)
        columns = np.sum(self.mat, axis=0)
        recall = [diagonals[i] / columns[i] for i in range(n[0])]
        return np.array(recall)