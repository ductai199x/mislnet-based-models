import torch
from lightning.pytorch import LightningModule
from ..forensic_attention_model import FAM
from typing import Dict, Any


class FamPLWrapper(LightningModule):
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
        self.model = FAM(**model_config)
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
            torch.empty(1, 3, self.model.fe.patch_size, self.modl.fe.patch_size),
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        patches, sim_mat_gt = batch
        sim_mat_pred = self.model(patches)
        loss = ((sim_mat_pred - sim_mat_gt).pow(2).sum() / patches.shape[0]) * 1.0e-2
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        patches, sim_mat_gt = batch
        sim_mat_pred = self.model(patches)
        loss = ((sim_mat_pred - sim_mat_gt).pow(2).sum() / patches.shape[0]) * 1.0e-2
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)