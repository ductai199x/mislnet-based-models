import os
import re
import torch
import torch.nn as nn


def load_model_state_dict(model: nn.Module, state_dict, module_name=""):
    curr_model_state_dict = model.state_dict()
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
        print(f"Success! All necessary keys for module '{model.__class__.__name__}' are loaded!")
    else:
        not_loaded_keys = [k for k, v in curr_model_keys_status.items() if not v]
        print(f"Warning! Some keys are not loaded! Not loaded keys are:\n{not_loaded_keys}")
        if len(outstanding_keys) > 0:
            print(f"Outstanding keys are: {outstanding_keys}")
    model.load_state_dict(curr_model_state_dict, strict=False)


def get_available_models():
    return ["mislnet", "mislnet_v2", "mislnet_v3", "mislnet_v4"]


def get_base_model(model_name, model_config, model_ckpt=None):
    model_name = model_name.lower()
    available_models = get_available_models()
    assert (
        model_name in available_models
    ), f"Model {model_name} not available. Available models: {available_models}"

    if model_name == "mislnet":
        from .mislnet import MISLNet

        model = MISLNet(**model_config)
    elif model_name == "mislnet_v2":
        from .mislnet_v2 import MISLNet_v2

        model = MISLNet_v2(**model_config)
    elif model_name == "mislnet_v3":
        from .mislnet_v3 import MISLNet_v3

        model = MISLNet_v3(**model_config)
    elif model_name == "mislnet_v4":
        from .mislnet_v4 import MISLNet_v4

        model = MISLNet_v4(**model_config)

    if model_ckpt is not None:
        if os.path.exists(model_ckpt):
            ckpt = torch.load(model_ckpt, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            load_model_state_dict(model, state_dict)
        else:
            print(
                f"WARNING!!!!!! Model checkpoint/state_dict at '{model_ckpt}' does not exist! Skipping loading checkpoint!"
            )

    return model


def get_base_model_cls(model_name):
    model_name = model_name.lower()
    available_models = get_available_models()
    assert (
        model_name in available_models
    ), f"Model {model_name} not available. Available models: {available_models}"

    if model_name == "mislnet":
        from .mislnet import MISLNet

        cls = MISLNet
    elif model_name == "mislnet_v2":
        from .mislnet_v2 import MISLNet_v2

        cls = MISLNet_v2
    elif model_name == "mislnet_v3":
        from .mislnet_v3 import MISLNet_v3

        cls = MISLNet_v3
    elif model_name == "mislnet_v4":
        from .mislnet_v4 import MISLNet_v4

        cls = MISLNet_v4
    
    return cls
