import re
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from .layers import DenseBlock
from .fsm import FSM
from typing import *


class FSG(LightningModule):
    def __init__(self, model_config, **kwargs,):
        self.model = FSM(**model_config)
        self.model.eval()

    def forward(self, x1, x2):
        return self.model(x1, x2)
    
    
