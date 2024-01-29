from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self):
        super(Generator, self).__init__()

        self.DeCon = nn.Sequential(
            nn.ConvTranspose2d(1, 4, 1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU())





    # nn.BatchNorm2d(1),
    # nn.ReLU())

    def forward(self, raw):
        x = self.DeCon(raw)
        x = self.conv1(x)

        return x

generator = Generator()
x = torch.randn(2,1,448,448)
output = generator(x)
print(output.shape)


