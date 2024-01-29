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
                    nn.Conv2d(4, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Decoder = nn.Sequential(
                    nn.Conv2d(640, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())


        self.Decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Decoder4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Decoder5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.Decoder6 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Sigmoid())

    def forward(self, raw):
        x = self.DeCon(raw)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat((x1, x2), 1))
        x4 = self.conv4(torch.cat((x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x1, x2, x3, x4), 1))
        x = torch.cat((x1, x2, x3, x4, x5), 1)

        x = self.Decoder(x)
        x = self.Decoder1(torch.cat((x, x5), 1))
        x = self.Decoder2(torch.cat((x, x4), 1))
        x = self.Decoder3(torch.cat((x, x3), 1))
        x = self.Decoder4(torch.cat((x, x2), 1))
        x = self.Decoder5(torch.cat((x, x1), 1))
        x = self.Decoder6(x)
        return x

# class Generator(nn.Module):
#     """docstring for Generator"""
#
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.DeCon = nn.Sequential(
#             nn.ConvTranspose2d(1, 4, 1, 1))
#
#         self.conv1 = nn.Sequential(
#                     nn.Conv2d(4, 128, 3, 1, 1),
#                     nn.BatchNorm2d(128),
#                     nn.ReLU())
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(384, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(512, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#
#         self.Decoder1 = nn.Sequential(
#                     nn.Conv2d(640, 1024, 3, 1, 1),
#                     nn.BatchNorm2d(1024),
#                     nn.ReLU())
#
#         self.Decoder2 = nn.Sequential(
#             nn.Conv2d(1024, 512, 3, 1, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.Decoder3 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#
#         self.Decoder4 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#
#         self.Decoder5 = nn.Sequential(
#             nn.Conv2d(128, 3, 3, 1, 1),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid())
#
#     def forward(self, raw):
#         x = self.DeCon(raw)
#
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(torch.cat((x1, x2), 1))
#         x4 = self.conv4(torch.cat((x1, x2, x3), 1))
#         x5 = self.conv5(torch.cat((x1, x2, x3, x4), 1))
#         x = torch.cat((x1, x2, x3, x4, x5), 1)
#
#         x = self.Decoder1(x)
#         x = self.Decoder2(x)
#         x = self.Decoder3(x)
#         x = self.Decoder4(x)
#         x = self.Decoder5(x)
#         return x
