import torch.nn as nn

from .layers import ConvBlock, DenseBlock, ConstrainedConv
from .cam_id_base import CamIdBase
from typing import *


class MISLNet(CamIdBase):
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
        **args,
    ):
        super().__init__(patch_size, num_classes)
        self.variant = variant
        self.chosen_arch = self.arch[variant]
        self.num_filters = num_filters

        self.constrained_conv = ConstrainedConv(num_filters=num_filters)

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

        if self.num_classes > 0:
            self.output = nn.Linear(self.chosen_arch[-1][2], self.num_classes)

    def forward(self, x):
        x = self.constrained_conv(x)
        x = self.conv_blocks(x)
        x = x.flatten(1, -1)
        x = self.fc_blocks(x)
        if self.num_classes > 0:
            x = self.output(x)
        return x
