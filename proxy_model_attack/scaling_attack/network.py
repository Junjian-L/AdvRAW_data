from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.conv3(x2)
        x4 = self.conv4(x)
        output = self.relu(x3 + x4)

        return output




class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self):
        super(Generator, self).__init__()

        self.DeCon = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.res1 = nn.Sequential(
            Residual(16, 32)
        )

        self.res2 = nn.Sequential(
            Residual(32, 64)
        )

        self.res3 = nn.Sequential(
            Residual(64, 128)
        )

        self.res4 = nn.Sequential(
            Residual(128, 256)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            )

        self.DeConv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.DeConv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.DeConv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.DeConv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.DeConv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid())


    def forward(self, raw):
        x = self.DeCon(raw)
        x1 = self.conv1(x)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.res4(x4)

        x = self.conv2(x5)

        x = self.DeConv1(torch.cat((x, x5), 1))
        x = self.DeConv2(torch.cat((x, x4), 1))
        x = self.DeConv3(torch.cat((x, x3), 1))
        x = self.DeConv4(torch.cat((x, x2), 1))
        x = self.DeConv5(torch.cat((x, x1), 1))

        x = self.conv3(x)
        return x







