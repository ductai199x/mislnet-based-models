import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_classes: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
