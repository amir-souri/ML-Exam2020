from trainers import PyTorchTrainer
from torch import optim
from networks import *
from torchvision import transforms

MLP_0 = MLP_Net0()
mlp0 = PyTorchTrainer(nn_module=MLP_0, transform=transforms.ToTensor(),
                      optimizer=optim.SGD(MLP_0.parameters(), lr=0.01),
                      batch_size=128)
mlp0.train(epochs=111)
mlp0.save()

MLP_1 = MLP_Net1()
mlp1 = PyTorchTrainer(nn_module=MLP_1, transform=transforms.ToTensor(),
                     optimizer=optim.SGD(MLP_1.parameters(), lr=0.01),
                     batch_size=128)
mlp1.train(epochs=111)
mlp1.save()

CNN_0 = CNN_Net0()
cnn0 = PyTorchTrainer(nn_module=CNN_0, transform=transforms.ToTensor(),
                     optimizer=optim.SGD(CNN_0.parameters(), lr=0.01),
                     batch_size=128)
cnn0.train(epochs=111)
cnn0.save()

CNN_1 = CNN_Net1()
cnn1 = PyTorchTrainer(nn_module=CNN_1, transform=transforms.ToTensor(),
                     optimizer=optim.SGD(CNN_1.parameters(), lr=0.01),
                     batch_size=128)
cnn1.train(epochs=111)
cnn1.save()

CNN_Regu = CNN_Net_Regularisation()
cnnRegu = PyTorchTrainer(nn_module=CNN_Regu, transform=transforms.ToTensor(),
                         optimizer=optim.SGD(CNN_Regu.parameters(), lr=0.01),
                         batch_size=128)
cnnRegu.train(epochs=111, patience=3)
cnnRegu.save()
