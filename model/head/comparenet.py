import torch
import torch.nn as nn
from typing import Literal


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


class CompareNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=64):
        super().__init__()
        self.fc1 = DenseBlock(input_dim, hidden_dim, "relu")
        self.fc2 = DenseBlock(hidden_dim * 3, output_dim, "relu")
        self.fc3 = nn.Linear(output_dim, 2)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x = torch.cat((x1, x1 * x2, x2), dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
