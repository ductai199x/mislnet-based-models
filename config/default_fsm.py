from importlib.machinery import SourceFileLoader
from model.fsm import FSM

def config(fe_config, fe_ckpt=None):
    if isinstance(fe_config, str):
        fe_config_module = SourceFileLoader("config", fe_config).load_module()
        fe_config = fe_config_module.config
    return {
        "model_args": {
            "_target_": FSM,
            "fe_config": fe_config["model_args"],
            "fe_ckpt": fe_ckpt,
            "comparenet_config": {
                "hidden_dim": 2048,
                "output_dim": 64,
            },
        },
        "data_args": {
            "patch_size": 256,
            "calc_class_weights": True,
            "ignore_classes": [11, 13, 23, 24, 37, 42, 45],
            "remap_labels": True,
            "lookahead_size": 1024,
            "dissim_vs_sim_ratio": 1,
        },
        "training_args": {
            "max_epochs": 10,
            "batch_size": 64,
            "num_workers": 16,
            "accum_grad_batches": 1,
            "lr": 1.0e-3,
            "decay_step": 40000,
            "decay_rate": 0.80,
            "unfreeze_fe_step": 120_000,
        },
    }