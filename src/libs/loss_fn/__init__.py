from logging import getLogger
from typing import List, Optional

import torch
import torch.nn as nn

from src.libs.loss_fn.action_segmentation_loss import ActionSegmentationLoss

from ..dataset_csv import DATASET_CSVS
from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    use_class_weight: bool = False,
    weights: List[int] = None,
    dataset_name: Optional[str] = None,
    device: Optional[str] = None,
    ce: bool = True,
    focal: bool = True,
    tmse: bool = False,
    gstmse: bool = False,
    threshold: float = 4,
    ignore_index: int = 255,
    ce_weight: float = 1.0,
    focal_weight: float = 1.0,
    tmse_weight: float = 0.15,
    gstmse_weight: float = 0.15,
) -> nn.Module:

    if use_class_weight:
        if dataset_name is None:
            message = "dataset_name used for training should be specified."
            logger.error(message)
            raise ValueError(message)

        if device is None:
            message = "you should specify a device when you use class weight."
            logger.error(message)
            raise ValueError(message)

        if dataset_name not in DATASET_CSVS:
            message = "dataset_name is invalid."
            logger.error(message)
            raise ValueError(message)

        train_csv_file = DATASET_CSVS[dataset_name].train
        class_weight = get_class_weight(train_csv_file).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        ce_weight = torch.FloatTensor(ce_weight)
        if ce is True and (focal or tmse or gstmse) is False:
            if ce_weight is not None:
                criterion = nn.CrossEntropyLoss(weight=ce_weight).to(device)
            else:
                criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = ActionSegmentationLoss(
                ce=ce,
                focal=focal,
                tmse=tmse,
                gstmse=gstmse,
                weight=ce_weight,
                threshold=threshold,
                ignore_index=ignore_index,
                ce_weight=ce_weight,
                focal_weight=focal_weight,
                tmse_weight=tmse_weight,
                gstmse_weight=gstmse_weight,
                device=device,
            )
        logger.info("weights: {}".format(weights))
        logger.info("criterion: {}".format(criterion))

    return criterion
