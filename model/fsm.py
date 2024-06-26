import re
import torch
import torch.nn as nn

from .layers import DenseBlock
from .cam_id_base import CamIdBase
from .mislnet import MISLNet
from typing import *


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


class FSM(nn.Module):
    default_fe_config = {
        "_target_": MISLNet,
        "patch_size": 256,
        "variant": "p256",
    }

    def __init__(
        self,
        fe_config=default_fe_config,
        fe_ckpt=None,
        comparenet_config=dict(),
        **kwargs,
    ):
        super().__init__()
        fe_config["num_classes"] = 0  # to make fe without final classification layer
        self.fe: CamIdBase = self.load_module_from_ckpt(fe_config["_target_"], fe_ckpt, "", **fe_config)
        comparenet_config["input_dim"] = self.fe.chosen_arch[-1][2]
        self.comparenet = CompareNet(**comparenet_config)
        self.fe_freeze = True

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
        self.load_module_state_dict(self.comparenet, state_dict, module_name="comparenet")

    def forward(self, x1, x2):
        if self.fe_freeze:
            self.fe.eval()
            with torch.no_grad():
                x1 = self.fe(x1)
                x2 = self.fe(x2)
        else:
            self.fe.train()
            x1 = self.fe(x1)
            x2 = self.fe(x2)
        return self.comparenet(x1, x2)
