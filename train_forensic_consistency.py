import os
from importlib.machinery import SourceFileLoader
import rich
import argparse
import pandas as pd
import wandb

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from typing import *

from model.forensic_consistency_plwrapper import ForensicConsistencyPLWrapper
from data.img_consistency_dataset import ImageConsistencyDataset

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
EXPERIMENT_NAME = "frs_con_net"


def prepare_model(args: dict[str, Any]) -> ForensicConsistencyPLWrapper:
    state_dict = None
    if args["prev_ckpt"]:
        if args["resume"]:
            return ForensicConsistencyPLWrapper.load_from_checkpoint(args["prev_ckpt"])
        else:
            ckpt = torch.load(args["prev_ckpt"], map_location="cpu")
            state_dict = ckpt["state_dict"]
    model = ForensicConsistencyPLWrapper(args["model_args"], args["training_args"])
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def prepare_datasets(args: dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_metadata, val_metadata = [], []
    for data_dir in args["data_args"]["data_dirs"]:
        metadata = pd.read_csv(f"{data_dir}/metadata.csv", index_col=None)
        metadata["path"] = metadata["path"].apply(lambda x: f"{data_dir}/{x}")
        train_metadata.append(metadata.query("split == 'train'"))
        val_metadata.append(metadata.query("split == 'val'"))
    train_metadata = pd.concat(train_metadata).reset_index(drop=True)
    print("Limiting the number of samples to 5,000 for validation set")
    val_metadata = pd.concat(val_metadata).sample(n=5_000, random_state=42).reset_index(drop=True)

    args["data_args"].pop("data_dirs")
    train_ds = ImageConsistencyDataset(
        root_dir="",
        image_paths=train_metadata["path"].tolist(),
        **args["data_args"],
    )
    args["data_args"].update({"is_augment_input": False})
    val_ds = ImageConsistencyDataset(
        root_dir="",
        image_paths=val_metadata["path"].tolist(),
        **args["data_args"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args["training_args"]["batch_size"],
        shuffle=True,
        num_workers=args["training_args"]["num_workers"],
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args["training_args"]["batch_size"],
        shuffle=True,
        num_workers=args["training_args"]["num_workers"],
        persistent_workers=True,
    )
    return train_loader, val_loader


def prepare_logger(args: dict[str, Any]) -> Tuple[Optional[Union[WandbLogger, TensorBoardLogger]], str]:
    if args["fast_dev_run"]:
        return None, None

    if args.get("logger") is None:
        logger_method = "tensorboard"
    else:
        logger_method = args["logger"]
    args_log_dir = args["log_dir"]
    args_version = args["version"]
    args_uid = args["uid"]

    if logger_method == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f"version_{args_version}",
            name=args_log_dir,
            log_graph=True,
        )
        log_path = f"{args_log_dir}/version_{args_version}"
        return logger, log_path
    elif logger_method == "wandb":
        log_path = f"{args_log_dir}/version_{args_version}"
        wandb_path = f"{log_path}/wandb"
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        run_uid = args_uid if args_uid else wandb.util.generate_id()
        logger = WandbLogger(
            project=f"{EXPERIMENT_NAME}",
            save_dir=log_path,
            version=f"version_{args_version}_{run_uid}",
            name=f"{EXPERIMENT_NAME}_{args_version}_{run_uid}",
            log_model="all",
            resume="allow" if args["resume"] else "never",
        )
        return logger, wandb_path
    else:
        raise NotImplementedError(f"Unknown logger method: {logger_method}")


def train(args: argparse.Namespace) -> None:
    # define how the model is loaded in the prepare_model.py file
    train_dl, val_dl = prepare_datasets(args.__dict__)
    model = prepare_model(args.__dict__)
    logger, log_path = prepare_logger(args.__dict__)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckpt = ModelCheckpoint(
        dirpath=f"{log_path}/checkpoints",
        monitor="val_loss",
        filename=f"{args.pre + '-' if args.pre != '' else ''}{{epoch:02d}}-{{val_loss:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=5,
        mode="min",
    )
    callbacks = [] if args.fast_dev_run else [ModelSummary(-1), TQDMProgressBar(refresh_rate=1), model_ckpt, lr_monitor]

    if args.fast_dev_run:
        num_gpus = 1
    else:
        num_gpus = "auto" if args.gpus == -1 else args.gpus
    trainer = Trainer(
        accelerator="auto",
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=num_gpus,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.training_args.get("accum_grad_batches", 1),
        logger=logger,
        profiler=None,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=not args.fast_dev_run,
        log_every_n_steps=10,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=args.training_args.get("validation_interval", None),
    )
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="all", log_freq=100)
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.prev_ckpt if args.resume else None)


def parse_args(args: argparse.Namespace) -> argparse.Namespace:
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    if not os.path.exists(args.fe_config):
        raise FileNotFoundError(f"Feature extractor config file does not exist: {args.fe_config}")
    if args.prev_ckpt and not os.path.exists(args.prev_ckpt):
        raise FileNotFoundError(f"Previous checkpoint file does not exist: {args.prev_ckpt}")
    if args.log_dir and not os.path.isdir(args.log_dir):
        print(f"Log dir does not exist: {args.log_dir}. Trying to create it..")
        os.makedirs(args.log_dir)
    if args.resume and not args.prev_ckpt:
        raise ValueError("Resume is true but there's no checkpoint specified")


    # load the config file
    config_module = SourceFileLoader("config", args.config).load_module()
    config = config_module.config(args.fe_config, args.fe_ckpt)
    args.model_args = config["model_args"]
    args.data_args = config["data_args"]
    args.training_args = config["training_args"]
    args.max_epochs = args.training_args["max_epochs"]

    rich.print(args.__dict__)
    return args


def main():
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="the path to a config file",
        required=True,
    )
    parser.add_argument(
        "--fe-config",
        type=str,
        help="the path to a feature extractor config file",
        required=True,
    )
    parser.add_argument(
        "--fe-ckpt",
        type=str,
        help="the path to a feature extractor checkpoint file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--prev-ckpt",
        type=str,
        help="the path to a previous checkpoint",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume the training progress? False will reset the optimizer state (True/False)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="the version of this model (same as the one saved in log dir)",
        default="0",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        help="the path to the log directory",
        default=f"lightning_logs/{EXPERIMENT_NAME}",
    )
    parser.add_argument(
        "-f",
        "--fast-dev-run",
        action="store_true",
        help="fast dev run? (True/Fase)",
    )
    parser.add_argument(
        "--pre",
        type=str,
        help="checkpoint's prefix",
        default="",
    )
    parser.add_argument(
        "--gpus",
        type=lambda x: [int(i) for i in x.split(",")],
        help="specify which GPUs to use (comma-separated list) or leave it alone to use all available GPUs",
        default=-1,
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb"],
        help="logger method (tensorboard/wandb)",
        default="tensorboard",
    )
    parser.add_argument(
        "--uid",
        type=str,
        help="unique id for wandb in case resuming runs with same version",
        default=None,
    )
    args = parser.parse_args()
    args = parse_args(args)

    train(args)


if __name__ == "__main__":
    main()