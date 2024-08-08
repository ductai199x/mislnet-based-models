from importlib.machinery import SourceFileLoader
from model.mislnet import MISLNet

def config(fe_config, fe_ckpt=None):
    if isinstance(fe_config, str):
        fe_config_module = SourceFileLoader("config", fe_config).load_module()
        fe_config = fe_config_module.config
    return {
        "model_args": {
            "_target_": MISLNet,
            "fe_config": fe_config["model_args"],
            "fe_ckpt": fe_ckpt,
        },
        "data_args": {
            "patch_size": 128,
            "patch_per_dir": 5,
            "data_dirs": [
                "/media/nas2/misl_image_db",
                "/media/nas2/Datasets/unsplash_image_dataset",
                "/media/nas2/Datasets/pexel_image_dataset/",
            ],
            "is_augment_input": False,
        },
        "training_args": {
            "max_epochs": 100,
            "batch_size": 8,
            "num_workers": 8,
            "accum_grad_batches": 8,
            "lr": 5.0e-2,
            "decay_step": 4,
            "decay_rate": 0.80,
            "alpha": 1.0,
            "beta": 0.75,
            "gamma": 0.25,
        },
    }