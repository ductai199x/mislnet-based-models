import torch
import torch.nn as nn
from .base_model_class import BaseModel
from typing import *


class ConstrainedConv(nn.Module):
    def __init__(self, input_chan=3, num_filters=6, is_constrained=True):
        super().__init__()
        self.kernel_size = 5
        self.input_chan = input_chan
        self.num_filters = num_filters
        self.is_constrained = is_constrained
        weight = torch.empty(num_filters, input_chan, self.kernel_size, self.kernel_size)
        nn.init.xavier_normal_(weight, gain=1/3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.one_middle = torch.zeros(self.kernel_size * self.kernel_size)
        self.one_middle[12] = 1
        self.one_middle = nn.Parameter(self.one_middle, requires_grad=False)

    def forward(self, x):
        w = self.weight
        if self.is_constrained:
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


class MISLNet(BaseModel):
    arch = {
        "p256": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 6 * 6 * 128, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p256_3fc_256e": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 6 * 6 * 128, 1024, "tanh"),
            ("fc2", 1024, 512, "tanh"),
            ("fc3", 512, 256, "tanh"),
        ],
        "p128": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 2 * 2 * 128, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p96": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 8 * 4 * 64, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
        "p64": [
            ("conv1", -1, 96, 7, 2, "valid", "tanh"),
            ("conv2", 96, 64, 5, 1, "same", "tanh"),
            ("conv3", 64, 64, 5, 1, "same", "tanh"),
            ("conv4", 64, 128, 1, 1, "same", "tanh"),
            ("fc1", 2 * 4 * 64, 200, "tanh"),
            ("fc2", 200, 200, "tanh"),
        ],
    }

    def __init__(
        self,
        patch_size: int,
        variant: str,
        num_classes=0,
        num_filters=6,
        is_constrained=True,
        **kwargs,
    ):
        super().__init__(patch_size, num_classes)
        self.variant = variant
        self.chosen_arch = self.arch[variant]
        self.num_filters = num_filters

        self.constrained_conv = ConstrainedConv(num_filters=num_filters, is_constrained=is_constrained)

        self.conv_blocks = []
        self.fc_blocks = []
        for block in self.chosen_arch:
            if block[0].startswith("conv"):
                self.conv_blocks.append(
                    ConvBlock(
                        in_chans=(num_filters if block[1] == -1 else block[1]),
                        out_chans=block[2],
                        kernel_size=block[3],
                        stride=block[4],
                        padding=block[5],
                        activation=block[6],
                    )
                )
            elif block[0].startswith("fc"):
                self.fc_blocks.append(
                    DenseBlock(
                        in_chans=block[1],
                        out_chans=block[2],
                        activation=block[3],
                    )
                )

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.fc_blocks = nn.Sequential(*self.fc_blocks)

        self.register_buffer("flatten_index_permutation", torch.tensor([0, 1, 2, 3], dtype=torch.long))

        if self.num_classes > 0:
            self.output = nn.Linear(self.chosen_arch[-1][2], self.num_classes)

    def forward(self, x):
        x = self.constrained_conv(x)
        x = self.conv_blocks(x)
        x = x.permute(*self.flatten_index_permutation)
        x = x.flatten(1, -1)
        x = self.fc_blocks(x)
        if self.num_classes > 0:
            x = self.output(x)
        return x
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "flatten_index_permutation" not in state_dict:
            super().load_state_dict(state_dict, False, assign)
        else:
            super().load_state_dict(state_dict, strict, assign)
