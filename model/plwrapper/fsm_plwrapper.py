import re
import math
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from ..fsm import FSM
from typing import Dict, Any


class FsmPLWrapper(LightningModule):
    default_trainining_config = {
        "lr": 1.0e-3,
        "decay_step": 40000,
        "decay_rate": 0.80,
        "class_weights": None,
        "unfreeze_fe_step": None,
    }
    default_model_config = {}

    def __init__(
        self,
        model_config: Dict[str, Any] = default_model_config,
        training_config: Dict[str, Any] = default_trainining_config,
        **kwargs,
    ):
        super().__init__()
        self.model = FSM(**model_config)
        self.is_constr_conv = hasattr(self.model.fe, "constrained_conv")
        self.training_config = training_config
        self.class_weights = training_config.get("class_weights", self.default_trainining_config["class_weights"])
        self.unfreeze_fe_step = training_config.get("unfreeze_fe_step", self.default_trainining_config["unfreeze_fe_step"])

        self.train_acc = MulticlassAccuracy(num_classes=2)
        self.val_acc = MulticlassAccuracy(num_classes=2)
        self.test_acc = MulticlassAccuracy(num_classes=2)

        self.save_hyperparameters("model_config", "training_config")
        self.example_input_array = (
            torch.empty(1, 3, self.model.fe.patch_size, self.model.fe.patch_size),
            torch.empty(1, 3, self.model.fe.patch_size, self.model.fe.patch_size),
        )

        self.model.fe_freeze = True

    def modify_legacy_state_dict(self, state_dict):
        modified_state_dict = dict(state_dict)
        for k, v in state_dict.items():
            new_k = ""
            if "mislnet" in k:
                new_k += ".fe."
                if "weights_cstr" in k:
                    new_k += "constrained_conv.weight"
                elif "conv" in k:
                    block_num = int(re.search(r"(?<=conv)\d+", k).group(0)) - 1
                    type_ = re.search(r".+(?:\.)(.+)$", k).group(1)
                    new_k += f"conv_blocks.{block_num}.conv.{type_}"
                elif "bn" in k:
                    block_num = int(re.search(r"(?<=bn)\d+", k).group(0)) - 1
                    type_ = re.search(r".+(?:\.)(.+)$", k).group(1)
                    new_k += f"conv_blocks.{block_num}.bn.{type_}"
                elif "fc" in k:
                    block_num = int(re.search(r"(?<=fc)\d+", k).group(0)) - 1
                    type_ = re.search(r".+(?:\.)(.+)$", k).group(1)
                    new_k += f"fc_blocks.{block_num}.fc.{type_}"
            elif "comparenet" in k:
                new_k += ".comparenet."
                block_num = int(re.search(r"(?<=fc)\d+", k).group(0))
                type_ = re.search(r".+(?:\.)(.+)$", k).group(1)
                if block_num in [1, 2]:
                    new_k += f"fc{block_num}.fc.{type_}"
                elif block_num == 3:
                    new_k += f"fc{block_num}.{type_}"

            modified_state_dict[new_k] = v
            del modified_state_dict[k]
        modified_state_dict["fe.flatten_index_permutation"] = torch.LongTensor([0, 2, 3, 1])
        return modified_state_dict

    def on_load_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:
        if checkpoint.get("legacy") == True:
            checkpoint["state_dict"] = self.modify_legacy_state_dict(checkpoint["state_dict"])
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.model.load_state_dict(state_dict, strict, assign)

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat_1_2 = self.model(x1, x2)
        y_hat_2_1 = self.model(x2, x1)
        loss = F.cross_entropy(y_hat_1_2, y, weight=self.class_weights.to(self.device))
        loss += F.cross_entropy(y_hat_2_1, y, weight=self.class_weights.to(self.device))
        if self.unfreeze_fe_step is not None and self.global_step > self.unfreeze_fe_step:
            if self.model.fe_freeze:
                self.model.fe_freeze = False
            if self.is_constr_conv:
                if self.model.fe.constrained_conv.weight.std() > 1.0:
                    loss += min(
                        0.1, math.sqrt(1 / (self.global_step / 2000))
                    ) * self.model.fe.constrained_conv.weight.norm(2)

        self.train_acc(y_hat_1_2, y)
        self.train_acc(y_hat_2_1, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.model(x1, x2)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
        self.val_acc(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.model(x1, x2)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_config["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.training_config["decay_step"], gamma=self.training_config["decay_rate"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
