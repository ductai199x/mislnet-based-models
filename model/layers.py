import torch
import torch.nn as nn

from typing import *


class ConstrainedConv(nn.Module):
    def __init__(self, input_chan=3, num_filters=6):
        super().__init__()
        self.kernel_size = 5
        self.input_chan = input_chan
        self.num_filters = num_filters
        self.weight = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(num_filters, input_chan, self.kernel_size, self.kernel_size), gain=1 / 3
            ),
            requires_grad=True,
        )
        self.one_middle = torch.zeros(self.kernel_size * self.kernel_size)
        self.one_middle[12] = 1
        self.one_middle = nn.Parameter(self.one_middle, requires_grad=False)

    def forward(self, x):
        w = self.weight
        if self.training:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            w = w - w.mean(1)[..., None] + 1 / (self.kernel_size * self.kernel_size - 1)
            w = w - (w + 1) * self.one_middle
            w = w.view(self.num_filters, self.input_chan, self.kernel_size, self.kernel_size)
        x = nn.functional.conv2d(x, w, padding="valid")
        x = nn.functional.pad(x, (2, 3, 2, 3))
        return x


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        stride,
        padding,
        activation: Literal["tanh", "relu"],
    ):
        super().__init__()
        assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
        self.conv = torch.nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm2d(out_chans)
        self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        activation: Literal["tanh", "relu"],
    ):
        super().__init__()
        assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
        self.fc = torch.nn.Linear(in_chans, out_chans)
        self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))