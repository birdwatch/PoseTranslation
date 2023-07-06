import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Any, Dict

import torch
import torch.optim as optim
import wandb
from torchvision.transforms import Compose

from datasets import get_dataloader
from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_cls2id_map
from libs.config import get_config, save_config
from libs.device import get_device
from libs.helper import evaluate, train
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.mean_std import get_mean, get_std
from libs.seed import set_seed
from libs.transformer import Normalize, ToTensor
from models import get_model

# from torchvision.transforms import (
#     ColorJitter,
#     Compose,
#     Normalize,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     ToTensor,
# )


logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    parser.add_argument("--config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (Default: 42)",
    )

    return parser.parse_args()


def prepare_train(config: Dict[str, Any], device: str) -> tuple:
    # Dataloader
    normailzer = Normalize(mean=get_mean(), std=get_std())
    train_transform = Compose(
        [
            ToTensor(),
            normailzer,
        ]
    )

    val_transform = Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])

    train_loader = get_dataloader(config, "train", train_transform)

    val_loader = get_dataloader(config, "val", val_transform)

    # # the number of classes
    # n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(config, device)

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")

    # criterion for loss
    criterion = get_criterion(
        config.LOSS.USE_TARGET_WEIGHT,
        config.LOSS.WEIGHTS,
        config.DATASET.NAME,
        device,
        config.LOSS.CE,
        config.LOSS.FOCAL,
        config.LOSS.TMSE,
        config.LOSS.GSTMSE,
        config.LOSS.THRESHOLD,
        config.LOSS.IGNORE_INDEX,
        config.LOSS.CE_WEIGHT,
        config.LOSS.FOCAL_WEIGHT,
        config.LOSS.TMSE_WEIGHT,
        config.LOSS.GSTMSE_WEIGHT,
    )

    return (
        train_loader,
        val_loader,
        model,
        optimizer,
        begin_epoch,
        best_loss,
        criterion,
    )


def train_with_sweep(default_config, device) -> None:
    with wandb.init(config=default_config.to):
        config = wandb.config
        wandb.run.name = f"{config.MODEL.NAME}_{config.DATASET.NAME}_{config.TRAIN.optimizer}"
        wandb.run.save()

        (
            train_loader,
            val_loader,
            model,
            optimizer,
            begin_epoch,
            best_loss,
            criterion,
        ) = prepare_train(config, device)

        wandb.watch(model)

        result_path = os.path.join(os.path.dirname(config.result_root_dir), wandb.run.name)
        train_logger = TrainLogger(result_path)
        save_config(config, os.path.join(result_path, "config.yaml"))

        # training
        for epoch in range(begin_epoch, config.TRAIN.epochs):
            print("-" * 20)
            start = time.time()
            train_loss, train_acc1, train_f1s = train(train_loader, model, criterion, optimizer, epoch, device)
            train_time = int(time.time() - start)

            # validation
            start = time.time()
            val_loss, val_acc1, val_f1s, PCE, acc_per_phase = evaluate(val_loader, model, criterion, device)
            val_time = int(time.time() - start)

            # save a model if top1 acc is higher than ever
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(result_path, "best_model.prm"),
                )

            # save checkpoint every epoch
            save_checkpoint(result_path, epoch, model, optimizer, best_loss)

            # write logs to dataframe and csv file
            train_logger.update(
                epoch,
                optimizer.param_groups[0]["lr"],
                train_time,
                train_loss,
                train_acc1,
                train_f1s,
                val_time,
                val_loss,
                val_acc1,
                val_f1s,
                PCE,
                acc_per_phase,
            )

            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "train_acc@1": train_acc1,
                    "train_f1s": train_f1s,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                    "val_acc@1": val_acc1,
                    "val_f1s": val_f1s,
                    "PCE": PCE,
                    "acc_per_phase": acc_per_phase,
                },
                step=epoch,
            )

        # save models
        torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

        # delete checkpoint
        os.remove(os.path.join(result_path, "checkpoint.pth"))

        logger.info("Done")


def train_wo_sweep(config, device) -> None:
    (
        train_loader,
        val_loader,
        model,
        optimizer,
        begin_epoch,
        best_loss,
        criterion,
    ) = prepare_train(config, device)
    result_path = os.path.join(os.path.dirname(config.OUTPUT_DIR))
    log_path = os.path.join(result_path, "experiment.csv")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    train_logger = TrainLogger(log_path, False)
    save_config(config, os.path.join(result_path, "config.yaml"))

    # training
    for epoch in range(begin_epoch, config.TRAIN.EPOCHS):
        print("-" * 20)
        start = time.time()
        train_loss, train_acc1, train_f1s = train(train_loader, model, criterion, optimizer, epoch, device)
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_loss, val_acc1, val_f1s, PCE, acc_per_phase = evaluate(val_loader, model, criterion, device)
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizer.param_groups[0]["lr"],
            train_time,
            train_loss,
            train_acc1,
            train_f1s,
            val_time,
            val_loss,
            val_acc1,
            val_f1s,
            PCE,
            acc_per_phase,
        )


def main() -> None:
    args = get_arguments()

    # load config file
    default_config = get_config(args.config)

    # set logger
    if args.debug:
        basicConfig(level=DEBUG)
    else:
        basicConfig(level=INFO)

    # set seed
    set_seed(args.seed)

    # set device
    device = get_device()

    # Weights and biases
    if default_config.USE_SWEEP:
        train_with_sweep(default_config, device)  # for hyperparameter optimization
    else:
        train_wo_sweep(default_config, device)


if __name__ == "__main__":
    main()
