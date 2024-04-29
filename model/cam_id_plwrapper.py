import math
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from .cam_id_base import CamIdBase


class CamIdPLWrapper(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model: CamIdBase = model_config["_target_"](**model_config)
        self.is_constr_conv = hasattr(self.model, "constrained_conv")
        self.training_config = training_config
        self.class_weights = training_config["class_weights"]

        self.train_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.model.num_classes)

        self.save_hyperparameters()
        self.example_input_array = torch.randn(1, 3, self.model.patch_size, self.model.patch_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
        if self.is_constr_conv and loss < 1.0:
            loss += min(
                1, math.sqrt(1 / (self.global_step / 10000))
            ) * self.model.constrained_conv.weight.norm(2)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        if self.is_constr_conv:
            self.log(
                "constr_conv_std",
                self.model.constrained_conv.weight.std(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
        self.val_acc(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.test_acc(y_hat, y)
        self.test_confusion_matrix(y_hat, y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_confusion_matrix", self.test_confusion_matrix, on_step=False, on_epoch=True, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.training_config["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.training_config["decay_step"], gamma=self.training_config["decay_rate"]
        )

        return [optimizer], [lr_scheduler]
