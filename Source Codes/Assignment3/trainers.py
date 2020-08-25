import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from fashionmnist_utils import mnist_reader
from metrics import MetricLogger


class Trainer(ABC):
    """Provides a library-independent API for training and evaluating machine learning classifiers."""

    def __init__(self, model):
        """Creates a new model instance with a unique name and underlying model.

        :param model: Model object to be used in training/prediction/evaluation.
        """
        self.model = model
        self.name = f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'

    @abstractmethod
    def predict(self, input):
        ...

    @abstractmethod
    def train(self, *args):
        """Completely trains self.model using internal training data
        """
        ...

    @abstractmethod
    def evaluate(self) -> MetricLogger:
        """Evaluate model on the internal testing data.

        :returns: MetricLogger object with results.
        """
        ...

    @abstractmethod
    def save(self):
        """Save the model object in "models". The filename is given by self.name.
        """
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        """Load the model object at the specified file location.

        :param path: Path in "models" directory to load from.
        """
        ...


class SKLearnTrainer(Trainer):
    """Implements the Model API for scikit-learn models."""

    def __init__(self, algorithm):
        super().__init__(algorithm)
        X, y = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='train')
        # Load and split datasets into training, validation, and test set.
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test, self.y_test = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='t10k')

    def predict(self, input):
        return self.model.predict(input)

    def train(self):
        model_name = type(self.model).__name__
        since = datetime.now()
        start_time = since.strftime("%H:%M:%S")
        print(f"Training is started for {model_name} model at {start_time}")
        self.model.fit(self.X_train, self.y_train)
        time_elapsed = datetime.now() - since
        print(f"Training for {model_name} model completed in {time_elapsed}")
        pred = self.model.predict(self.X_val)
        obj = MetricLogger(one_hot=False)
        obj.log(pred, self.y_val)
        print(f"Accuracy of {model_name} model:", obj.accuracy, sep='\n')
        print(f"Precision of {model_name} model:", obj.precision, sep='\n')
        print(f"Recall of {model_name} model:", obj.recall, sep='\n')

    def evaluate(self):
        pred = self.model.predict(self.X_test)
        obj = MetricLogger(one_hot=False)
        obj.log(pred, self.y_test)
        return obj

    def save(self):
        with open(os.path.join('models', self.name + '.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load(path: str) -> Trainer:
        new = SKLearnTrainer(None)
        with open(path, 'rb') as file:
            new.model = pickle.load(file)
            new.name = os.path.basename(path).split('.')[0]
            return new


def get_data(transform, train=True):
    return FashionMNIST(os.path.join(os.getcwd(), 'data'), train=train, transform=transform, download=True)


class PyTorchTrainer(Trainer):
    """Implements the Model API for PyTorch (torch) models."""

    def __init__(self, nn_module: nn.Module, transform: Callable, optimizer: torch.optim.Optimizer, batch_size: int):
        """Initialize model.

        :param nn_module: torch.nn.Module to use for the model.
        :param transform: torchvision.transforms.Transform to apply to dataset images.
        :param optimizer: torch.optim.Optimizer
        :param batch_size: Batch size to use for datasets.
        """
        super().__init__(nn_module)

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Load datasets
        self.train_data, self.val_data, self.test_data = None, None, None
        self.init_data()

        # Create logger for TensorBoard
        self.logger = SummaryWriter()

    def init_data(self):
        """Method for loading datasets.
        """
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)

        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)

    def predict(self, input):
        return self.model(input).argmax(dim=1)

    def train(self, epochs: int, patience=None):
        model_name = type(self.model).__name__
        since = datetime.now()
        start_time = since.strftime("%H:%M:%S")
        print(f"Training and Validating is started for {model_name} at {start_time}")
        count = 0
        obj_train = MetricLogger(one_hot=True)
        obj_val = MetricLogger(one_hot=True)

        if patience is None:
            print("patience is None")
            for e in range(epochs):
                print('epoch', e)
                obj_train.reset()
                obj_val.reset()
                for i, (x, y) in enumerate(self.train_data):
                    preds_train = self.model.forward(x)
                    loss = F.cross_entropy(preds_train, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    obj_train.log(preds_train, y)
                    count += 1
                    if i % 100 == 0:
                        self.logger.add_scalar("train loss (batches)", loss.item(), count)
                        self.logger.add_scalar("train accuracy (batches)", obj_train.accuracy, count)
                for (x, y) in self.val_data:
                    preds_val = self.model.forward(x)
                    obj_val.log(preds_val, y)
                count += 1
                self.logger.add_scalar("validation accuracy (epochs)", obj_train.accuracy, count)
            time_elapsed = datetime.now() - since
            print(f"Training and Validating for {model_name} completed in {time_elapsed}")

        if patience is not None and patience > 0:
            losses = []
            for e in range(epochs):
                print('epoch', e)
                obj_train.reset()
                obj_val.reset()
                self.model.train()
                for i, (x, y) in enumerate(self.train_data):
                    preds_train = self.model.forward(x)
                    loss = F.cross_entropy(preds_train, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    obj_train.log(preds_train, y)
                    count += 1
                    if i % 100 == 0:
                        self.logger.add_scalar("train loss (batches)", loss.item(), count)
                        self.logger.add_scalar("train accuracy (batches)", obj_train.accuracy, count)
                self.model.eval()
                val_loss = 0
                for (x, y) in self.val_data:
                    preds_val_Regu = self.model.forward(x)
                    loss = F.cross_entropy(preds_val_Regu, y)
                    val_loss += loss
                    obj_val.log(preds_val_Regu, y)
                if len(losses) == 0 or val_loss < losses[-1]:
                    losses = [val_loss]
                else:
                    losses.append(val_loss)
                if len(losses) == patience:
                    print("Early stopping")
                    break
                count += 1
                self.logger.add_scalar("validation accuracy (epochs)", obj_train.accuracy, count)

            time_elapsed = datetime.now() - since
            print(f"Training and Validating for {model_name} completed in {time_elapsed}")

        if patience is not None and patience <= 0:
            time_elapsed = datetime.now() - since
            print(f"Training and Validating for {model_name} completed in {time_elapsed}")
            raise ValueError("patience must be positive!")

    def evaluate(self) -> MetricLogger:
        obj_test = MetricLogger(one_hot=True)
        self.model.eval()
        for (x, y) in self.test_data:
            preds_test = self.model.forward(x)
            obj_test.log(preds_test, y)

        return obj_test

    def save(self):
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join('models', self.name)
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> Trainer:
        with open(path, 'rb') as file:
            new = pickle.load(file)
            new.init_data()
            return new