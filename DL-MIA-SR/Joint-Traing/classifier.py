import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, x_dim, experiment):
        super(MLP, self).__init__()
        self.e = experiment
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.loss = []
        self.outputs = []
        self.use_cuda = self.e.config.use_cuda
        self.criterion = nn.CrossEntropyLoss()
        # self.dropout = nn.Dropout(p=0.6)

    def forward(self, vector):
        inputs = vector.view(-1, self.x_dim)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        outputs = F.softmax(self.fc3(inputs))

        return outputs

class reweightB(nn.Module):
    def __init__(self, experiment):
        super(reweightB, self).__init__()
        self.e = experiment
        self.x_dim = self.e.config.batch_size
        self.x_dimX = self.e.config.pair_size % self.e.config.batch_size
        self.fc1 = nn.Linear(self.x_dim, self.x_dim)
        self.fcx = nn.Linear(self.x_dimX, self.x_dimX)
        self.use_cuda = self.e.config.use_cuda
        self.dist = nn.CrossEntropyLoss()

    def forward(self, vector):
        if vector.shape[0] < self.e.config.batch_size:
            inputs = vector.view(-1, self.x_dimX)
            outputs = F.sigmoid(self.fcx(inputs))
        else:
            inputs = vector.view(-1, self.x_dim)
            outputs = F.sigmoid(self.fc1(inputs))

        return outputs


class reweight(nn.Module):
    def __init__(self, experiment):
        super(reweight, self).__init__()
        self.e = experiment
        self.x_dim = self.e.config.pair_size
        self.fc1 = nn.Linear(self.x_dim, self.x_dim)
        self.use_cuda = self.e.config.use_cuda
        self.dist = nn.CrossEntropyLoss()

    def forward(self, vector):
        inputs = vector.view(-1, self.x_dim)
        outputs = F.sigmoid(self.fc1(inputs))

        return outputs

