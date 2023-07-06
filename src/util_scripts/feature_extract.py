import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from datasets import get_dataloader

logger = logging.getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        prepare features for training.
        """
    )
    parser.add_argument("--config", type=str, help="path of a config file")


def feature_extract(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> np.ndarray:
    print("feature_extract")
    model.eval()
    features = []
    with torch.no_grad():
        for sample in dataloader:
            x = sample["images"].to(device)
            output = model(x)
            features.append(output.cpu().numpy())
    return np.concatenate(features, axis=0)
