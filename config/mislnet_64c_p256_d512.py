from model.base_model import MISLNet

config = {
    "model_args": {
        "_target_": MISLNet,
        "patch_size": 256,
        "variant": "p256",  # "p256_3fc_256e", "p256", "p128", "p96", "p64"
        "num_classes": 64,
        "num_filters": 6,
    },
    "data_args": {
        "patch_size": 512,
        "calc_class_weights": True,
        "ignore_classes": [11, 13, 23, 24, 37, 42, 45],
        "remap_labels": True,
    },
    "training_args": {
        "max_epochs": 70,
        "batch_size": 32,
        "num_workers": 16,
        "accum_grad_batches": 2,
        "lr": 2.0e-2,
        "decay_step": 3,
        "decay_rate": 0.75,
    },
}
