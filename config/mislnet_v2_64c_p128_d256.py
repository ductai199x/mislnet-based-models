from model.base_model import MISLNet_v2

config = {
    "model_args": {
        "_target_": MISLNet_v2,
        "patch_size": 128,
        "variant": "p128",  # "p256_3fc_256e", "p256", "p128", "p96", "p64"
        "num_classes": 64,
        "num_constrained_filters": 64,
        "num_rgb_filters": 64,
    },
    "data_args": {
        "patch_size": 256,
        "calc_class_weights": True,
        "ignore_classes": [11, 13, 23, 24, 37, 42, 45],
        "remap_labels": True,
    },
    "training_args": {
        "max_epochs": 70,
        "batch_size": 64,
        "num_workers": 16,
        "accum_grad_batches": 1,
        "lr": 1.0e-3,
        "decay_step": 3,
        "decay_rate": 0.50,
    },
}
