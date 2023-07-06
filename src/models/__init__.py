import sys
from logging import getLogger
from typing import Any, Dict

import torch.nn as nn

sys.path.append(".")

import src.models.phase_net
import src.models.swing_net
import src.models.swing_net_v2
import src.models.swing_net_v3

__all__ = ["get_model"]

logger = getLogger(__name__)


def get_model(config: Dict[str, Any], device: str) -> nn.Module:
    logger.info("{} will be used as a model.".format(config.MODEL.NAME))

    model = eval("src.models." + config.MODEL.NAME + ".get_model")(config, device)

    return model.to(device)
