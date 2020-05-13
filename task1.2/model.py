import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections
class smallVGG(nn.Module):
    def __init__(self, NChannels):
        super(smallVGG, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.ReLU12 = nn.ReLU()
        self.features.append(self.ReLU12)
        self.layerDict['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1


        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU()
        self.features.append(self.ReLU21)
        self.layerDict['ReLU21'] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)
        self.layerDict['conv31'] = self.conv31

        self.ReLU31 = nn.ReLU()
        self.features.append(self.ReLU31)
        self.layerDict['ReLU31'] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)
        self.layerDict['conv32'] = self.conv32


        self.ReLU32 = nn.ReLU()
        self.features.append(self.ReLU32)
        self.layerDict['ReLU32'] = self.ReLU32

        self.pool3 = nn.MaxPool2d(2,2)
        self.features.append(self.pool3)
        self.layerDict['pool3'] = self.pool3

        self.classifier = []

        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x