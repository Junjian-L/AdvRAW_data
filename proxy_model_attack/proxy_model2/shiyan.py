# Copyright 2020 by Andrey Ignatov. All Rights Reserved.
import torch

from torch.autograd import gradcheck
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.net(x)
        return y


net = Net()
net = net.double()
inputs = torch.randn((10, 15), requires_grad=True, dtype=torch.double)
targets = torch.randn((10, 1), dtype=torch.double)
targets2 = torch.randn((10, 1), dtype=torch.double)
outputs = net(inputs)
outputs2 = 5.0 * outputs
loss = F.mse_loss(outputs, targets) + 2.0 * F.mse_loss(outputs2, targets2)
loss.backward()
print(inputs.grad)




