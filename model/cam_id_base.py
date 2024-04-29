import torch.nn as nn


class CamIdBase(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_classes: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
