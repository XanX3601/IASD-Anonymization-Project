import torch
import torch.nn as nn
import torch.nn.functional as F


class Neural_Network_Classifier(nn.Module):
    def __init__(self):
        super(Neural_Network_Classifier, self).__init__()

        # Data
        # ---------------
        self.f = 3  # Number of features
        self.o = 1  # Number of output classes

        # Hyper parameters
        # ---------------
        self.b = False  # Bias
        self.blocks = 1  # Number of blocks for the resnet
        self.c = 16  # Number of filters

        # Convolution layers
        # ---------------
        self.conv_input = nn.Conv2d(self.f, self.c, (3, 3), 1, 1, bias=self.b)
        self.conv_main = nn.Conv2d(self.c, self.c, (3, 3), 1, 1, bias=self.b)

        # Normalization layers
        # ---------------
        self.batch_norm = nn.BatchNorm2d(self.c)

        # Pooling layers
        # ---------------
        self.avg_pool = nn.AvgPool2d((3, 3))

        # Dense layers
        # ---------------
        self.dense_out = nn.Linear(self.c * (10 ** 2), self.o, self.b)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.batch_norm(x)
        x = F.relu(x)

        for _ in range(self.blocks):
            residual = x
            x = self.conv_main(x)
            x = self.batch_norm(x)
            x = F.relu(x)
            x = self.conv_main(x)
            x = self.batch_norm(x)
            x = x + residual
            x = F.relu(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1, -1)
        x = self.dense_out(x)
        x = torch.sigmoid(x)
        return x

class Neural_Network_Meta_Classifier(nn.Module):
    def __init__(self, input_size):
        super(Neural_Network_Meta_Classifier, self).__init__()

        # Data
        # ---------------
        self.f = input_size # Size of the input vector
        self.o = 1  # Number of output classes

        # Dense layers
        # ---------------
        self.dense_in = nn.Linear(self.f, 128)
        self.dense_1 = nn.Linear(128, 256)
        self.dense_out = nn.Linear(256, self.o)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.dense_1(x)
        x = self.dense_out(x)
        x = torch.sigmoid(x)
        return x
