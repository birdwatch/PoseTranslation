from logging import getLogger

import torch.nn as nn
import torchvision

__all__ = ["get_model"]

model_names = ["resnet18", "resnet34", "resnet50"]
logger = getLogger(__name__)


def get_model(model_name: str, params: dict, device: str) -> nn.Module:
    model_name = model_name.lower()
    pretrained = params["pretrained"]
    if model_name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose resnet18, resnet34, resnet50 as a model."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(model_name))

    model = eval("models." + model_name + "models")(*params)

    return model.to(device)
