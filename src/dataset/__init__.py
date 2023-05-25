from logging import getLogger
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .flower_dataset import FlowersDataset

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    config: Dict[str, Any],
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    logger.info(
        f"Dataset: {config.DATASET.NAME}\tSplit: {config.DATASET.SPLIT}\tBatch size: {config.TRAIN.BATCH_SIZE}."
    )

    data = eval("models." + config.DATASET.NAME + ".get_dataset")(config)

    dataloader = DataLoader(
        data,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.TRAIN.WORKER,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader
