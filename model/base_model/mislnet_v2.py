import torch
import torch.nn as nn
from .base_model_class import BaseModel
from typing import *


class FusedConstrainedConv(nn.Module):
    def __init__(
        self, 
        num_constrained_filters=32,
        num_rgb_filters=32,
        input_chan=3,
    ):
        super().__init__()
        self.kernel_size = 5
        self.input_chan = input_chan
        self.num_constrained_filters = num_constrained_filters
        self.num_rgb_filters = num_rgb_filters
        self.output_chan = num_constrained_filters + num_rgb_filters

        weight = torch.empty(num_constrained_filters, input_chan, self.kernel_size, self.kernel_size)
        nn.init.xavier_normal_(weight, gain=1/3)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.one_middle = torch.zeros(self.kernel_size * self.kernel_size)
        self.one_middle[self.one_middle.numel() // 2] = 1
        self.one_middle = nn.Parameter(self.one_middle, requires_grad=False)

        rgb_weight = torch.empty(num_rgb_filters, input_chan, self.kernel_size, self.kernel_size)
        nn.init.xavier_normal_(rgb_weight, gain=1/3)
        self.rgb_weight = nn.Parameter(rgb_weight, requires_grad=True)

    def forward(self, x):
        w_cstr = self.weight
        w_cstr = w_cstr.view(-1, self.kernel_size * self.kernel_size)
        w_cstr = w_cstr - w_cstr.mean(1)[..., None] + 1 / (self.kernel_size * self.kernel_size - 1)
        w_cstr = w_cstr - (w_cstr + 1) * self.one_middle
        w_cstr = w_cstr.view(self.num_constrained_filters, self.input_chan, self.kernel_size, self.kernel_size)
        weight = torch.cat([w_cstr, self.rgb_weight], dim=0)
        x = nn.functional.conv2d(x, weight, padding="same")
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


class MISLNet_v2(BaseModel):
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
        num_constrained_filters=6,
        num_rgb_filters=32,
        **kwargs,
    ):
        super().__init__(patch_size, num_classes)
        self.variant = variant
        self.chosen_arch = self.arch[variant]
        self.num_constrained_filters = num_constrained_filters
        self.num_rgb_filters = num_rgb_filters

        self.constrained_conv = FusedConstrainedConv(num_constrained_filters, num_rgb_filters)
        

        self.conv_blocks = []
        self.fc_blocks = []
        for block in self.chosen_arch:
            if block[0].startswith("conv"):
                self.conv_blocks.append(
                    ConvBlock(
                        in_chans=(self.constrained_conv.output_chan if block[1] == -1 else block[1]),
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
