import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from model.mislnet import MISLNet
from typing import Type, Union, Dict, Any


class ForensicConsistencyPLWrapper(LightningModule):
    default_trainining_config = {
        "lr": 1.0e-2,
        "decay_step": 1,
        "decay_rate": 0.80,
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.1,
    }
    default_model_config = {
        "_target_": MISLNet,
        "fe_config": {
            "patch_size": 128,
            "variant": "p128",
            "num_classes": 0,
            "num_filters": 6,
            "is_constrained": True,
        },
        "fe_ckpt": None,
    }

    def __init__(
        self,
        model_config: Dict[str, Any] = default_model_config,
        training_config: Dict[str, Any] = default_trainining_config,
        **kwargs,
    ):
        super().__init__()
        model_cls = model_config.pop("_target_")
        model_config["fe_name"] = model_cls.__name__
        model_config["fe_config"]["num_classes"] = 0
        self.fe: nn.Module = self.load_module_from_ckpt(
            model_cls, model_config["fe_ckpt"], "", **model_config["fe_config"]
        )
        self.is_constr_conv = hasattr(self.fe, "constrained_conv")

        self.training_config = training_config

        self.alpha = training_config.get("alpha", self.default_trainining_config["alpha"])
        self.beta = training_config.get("beta", self.default_trainining_config["beta"])
        self.gamma = training_config.get("gamma", self.default_trainining_config["gamma"])

        self.save_hyperparameters(model_config, training_config)

        self.example_input_array = torch.empty(1, 3, self.fe.patch_size, self.fe.patch_size)

    def load_module_state_dict(self, module: nn.Module, state_dict, module_name=""):
        curr_model_state_dict = module.state_dict()
        curr_model_keys_status = {k: False for k in curr_model_state_dict.keys()}
        outstanding_keys = []
        for ckpt_layer_name, ckpt_layer_weights in state_dict.items():
            if module_name not in ckpt_layer_name:
                continue
            ckpt_matches = re.findall(r"(?=(?:^|\.)((?:\w+\.)*\w+)$)", ckpt_layer_name)[::-1]
            model_layer_name_match = list(set(ckpt_matches).intersection(set(curr_model_state_dict.keys())))
            # print(ckpt_layer_name, model_layer_name_match)
            if len(model_layer_name_match) == 0:
                outstanding_keys.append(ckpt_layer_name)
            else:
                model_layer_name = model_layer_name_match[0]
                assert (
                    curr_model_state_dict[model_layer_name].shape == ckpt_layer_weights.shape
                ), f"Ckpt layer '{ckpt_layer_name}' shape {ckpt_layer_weights.shape} does not match model layer '{model_layer_name}' shape {curr_model_state_dict[model_layer_name].shape}"
                curr_model_state_dict[model_layer_name] = ckpt_layer_weights
                curr_model_keys_status[model_layer_name] = True

        if all(curr_model_keys_status.values()):
            print(f"Success! All necessary keys for module '{module.__class__.__name__}' are loaded!")
        else:
            not_loaded_keys = [k for k, v in curr_model_keys_status.items() if not v]
            print(f"Warning! Some keys are not loaded! Not loaded keys are:\n{not_loaded_keys}")
            if len(outstanding_keys) > 0:
                print(f"Outstanding keys are: {outstanding_keys}")
        module.load_state_dict(curr_model_state_dict, strict=False)

    def load_module_from_ckpt(
        self,
        module_class: Type[nn.Module],
        ckpt_path: Union[None, str],
        module_name: str,
        *args,
        **kwargs,
    ) -> nn.Module:
        module = module_class(*args, **kwargs)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_state_dict = ckpt["state_dict"]
            self.load_module_state_dict(module, ckpt_state_dict, module_name=module_name)
        return module

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.load_module_state_dict(self.fe, state_dict, module_name="fe")

    def forward(self, x):
        pred = self.fe(x)
        pred = F.normalize(pred, p=2, dim=1)
        return pred @ pred.t()

    def build_label_and_weight_matrices(self, batch):
        device = batch.device
        label_block_mat = []
        loss_weight_block_mat = []
        for img in batch:
            num_cen, num_nei, _, _, _ = img.shape
            label_blk = torch.ones(num_cen * num_nei, num_cen * num_nei)
            loss_weight_blk = torch.block_diag(
                *[torch.ones(num_nei, num_nei) * (self.alpha - self.beta) for _ in range(num_cen)]
            )
            loss_weight_blk += self.beta - self.gamma
            label_block_mat.append(label_blk)
            loss_weight_block_mat.append(loss_weight_blk)

        label_block_mat = torch.block_diag(*label_block_mat)
        loss_weight_block_mat = torch.block_diag(*loss_weight_block_mat)
        loss_weight_block_mat += self.gamma
        return label_block_mat.to(device), loss_weight_block_mat.to(device)

    def loss_fn(self, pred_mat, label_mat, weight_mat):
        return (pred_mat - label_mat).pow(2).mul(weight_mat).sum().sqrt()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x = torch.cat(batch, dim=0)
        else:
            x = batch
        label_mat, weight_mat = self.build_label_and_weight_matrices(x)
        x = x.view(-1, *x.shape[-3:])
        pred_mat = self(x)
        loss = self.loss_fn(pred_mat, label_mat, weight_mat)
        if self.is_constr_conv:
            if self.fe.constrained_conv.weight.std() > 1.0:
                loss = loss + self.fe.constrained_conv.weight.norm(2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            x = torch.cat(batch, dim=0)
        else:
            x = batch
        label_mat, weight_mat = self.build_label_and_weight_matrices(x)
        x = x.view(-1, *x.shape[-3:])
        pred_mat = self(x)
        loss = self.loss_fn(pred_mat, label_mat, weight_mat)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.training_config["decay_step"], gamma=self.training_config["decay_rate"]
        )
        return [optimizer], [scheduler]
