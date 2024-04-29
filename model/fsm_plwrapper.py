import math
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from .fsm import FSM


class FsmPLWrapper(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model: FSM = model_config["_target_"](**model_config)
        self.is_constr_conv = hasattr(self.model.fe, "constrained_conv")
        self.training_config = training_config
        self.class_weights = training_config["class_weights"]

        self.train_acc = MulticlassAccuracy(num_classes=2)
        self.val_acc = MulticlassAccuracy(num_classes=2)
        self.test_acc = MulticlassAccuracy(num_classes=2)

        self.save_hyperparameters()
        self.example_input_array = (torch.empty(1, 3, self.model.fe.patch_size, self.model.fe.patch_size), torch.empty(1, 3, self.model.fe.patch_size, self.model.fe.patch_size))

        self.model.fe_freeze = True

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat_1_2 = self.model(x1, x2)
        y_hat_2_1 = self.model(x2, x1)
        loss = F.cross_entropy(y_hat_1_2, y, weight=self.class_weights.to(self.device))
        loss += F.cross_entropy(y_hat_2_1, y, weight=self.class_weights.to(self.device))
        if self.global_step > 800_000:
            if self.model.fe_freeze:
                self.model.fe_freeze = False
            if self.is_constr_conv:
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
