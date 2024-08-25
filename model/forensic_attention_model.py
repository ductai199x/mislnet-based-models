import re
import torch
import torch.nn as nn
from .head import ForensicAttentionHead
from .base_model import BaseModel, get_base_model_cls
from typing import Any, Type, Union


class FAM(nn.Module):
    """
    Forensic Attention Model (FAM) module.

    Args:
        fe_config (dict[str, Any]): Configuration dictionary for the feature extractor.
        fe_ckpt (str, optional): Path to the checkpoint file for the feature extractor. Defaults to None.
        freeze_fe (bool, optional): Whether to freeze the feature extractor during training. Defaults to True.
        num_heads (int, optional): Number of attention heads in the ForensicAttentionHead. Defaults to 8.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension in the ForensicAttentionHead. Defaults to 4.
        num_blocks (int, optional): Number of blocks in the ForensicAttentionHead. Defaults to 4.
        qkv_bias (bool, optional): Whether to include bias in the query, key, and value projections in the ForensicAttentionHead. Defaults to True.
    """
    
    def __init__(
        self,
        fe_config: dict[str, Any],
        fe_ckpt: str = None,
        freeze_fe: bool = True,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_blocks: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if "_target_" in fe_config:
            fe_cls = fe_config.pop("_target_")
            fe_config["model_name"] = fe_cls.__name__
        else:
            fe_cls = get_base_model_cls(fe_config["model_name"])
        fe_config["num_classes"] = 0  # to make fe without final classification layer
        self.fe: BaseModel = self.load_module_from_ckpt(fe_cls, fe_ckpt, "", **fe_config)
        self.freeze_fe = freeze_fe
        self.patch_size = self.fe.patch_size
        example_tensor = torch.randn((1, 3, self.patch_size, self.patch_size), requires_grad=False)
        self.fe_embed_dim = self.fe(example_tensor).shape[1]

        self.head = ForensicAttentionHead(
            self.fe_embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

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
        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        except Exception as e:
            print(f"Error loading state dict: {e}, trying to load manually")
            self.load_module_state_dict(self.fe, state_dict, module_name="fe")
            self.load_module_state_dict(self.head, state_dict, module_name="head")

    def forward_fe(self, x):
        if self.freeze_fe:
            self.fe.eval()
            with torch.no_grad():
                return self.fe(x)
        else:
            self.fe.train()
            return self.fe(x)

    def forward(self, x):
        B, P, C, H, W = x.shape
        x = x.view(B * P, C, H, W)
        x = self.forward_fe(x)
        x = x.view(B, P, -1)
        x = self.head(x)
        return x
