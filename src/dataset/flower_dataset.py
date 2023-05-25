from logging import getLogger
from typing import Any, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = getLogger(__name__)


class FlowersDataset(Dataset):
    def __init__(
        self, csv_file: str, transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.n_classes = self.df["class_id"].nunique()
        self.transform = transform

        logger.info(f"the number of classes: {self.n_classes}")
        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.df.iloc[idx]["image_path"]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        cls_id = self.df.iloc[idx]["class_id"]
        cls_id = torch.tensor(cls_id).long()

        label = self.df.iloc[idx]["label"]

        sample = {
            "img": img,
            "class_id": cls_id,
            "label": label,
            "img_path": img_path,
        }

        return sample

    def get_n_classes(self) -> int:
        return self.n_classes
