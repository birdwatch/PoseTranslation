import sys
from logging import getLogger
from typing import Any, Dict

from torch.utils.data import DataLoader

sys.path.append(".")

import src.datasets.golfdb
import src.datasets.golfdb_with_flow

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(config: Dict[str, Any], split: str, transform) -> DataLoader:
    """
    Args:
        config: configuration for training
        split: train or val
        Returns:
            dataloader: DataLoader for training
    """
    batch_size = config.TRAIN.BATCH_SIZE if split == "train" else 1
    shuffle = config.DATASET.SHUFFLE if split == "train" else False
    drop_last = config.DATASET.DROP_LAST if split == "train" else False

    logger.info(f"Dataset: {config.DATASET.NAME}\tSplit: {split}\tBatch size: {batch_size}.")

    data = eval("src.datasets." + config.DATASET.NAME + ".get_dataset")(config, split, transform)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=drop_last,
    )

    return dataloader
