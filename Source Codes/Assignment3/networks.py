import torch
import torch.nn.functional as F
from torch import nn


class MLP_Net0(nn.Module):
    def __init__(self):
        super(MLP_Net0, self).__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=21)
        self.linear2 = nn.Linear(in_features=21, out_features=10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class MLP_Net1(nn.Module):
    def __init__(self):
        super(MLP_Net1, self).__init__()
        self.linear1 = nn.Linear(in_features=784, out_features=85)
        self.linear2 = nn.Linear(in_features=85, out_features=10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class CNN_Net0(nn.Module):
    def __init__(self):
        super(CNN_Net0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4 * 1, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class CNN_Net1(nn.Module):
    def __init__(self):
        super(CNN_Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4 * 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

class CNN_Net_Regularisation(nn.Module):
    def __init__(self):
        super(CNN_Net_Regularisation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4 * 1, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout2d(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.relu(self.fc2(x))