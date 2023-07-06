import os
from functools import partial
from logging import getLogger
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset

logger = getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        if not os.path.exists(root_dir):
            raise FileNotFoundError("root directory not found.")

        self.root_dir: str = root_dir
        self.df = self._read_annotation_file(annotation_file)
        self.transform: Optional[transforms.Compose] = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("please implement this method.")

    def _read_annotation_file(self, annotation_file: str) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        try:
            if os.path.splitext(annotation_file)[1] == ".csv":
                df = pd.read_csv(annotation_file)
            elif os.path.splitext(annotation_file)[1] == ".pkl":
                df = pd.read_pickle(annotation_file)
            elif os.path.splitext(annotation_file)[1] == ".json":
                df = pd.read_json(annotation_file)
            else:
                raise NotImplementedError("annotation file format is not supported.")
        except BaseException as e:
            logger.exception(f"{e}")

        return df

    def _get_frames_from_video(self, vid_path: str, start_frame: int, end_frame: int) -> List[np.ndarray]:
        vid_cap = cv2.VideoCapture(vid_path)
        frames = []
        ret = True

        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while ret:
            ret, frame = vid_cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if vid_cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
                break
        vid_cap.release()

        frames = np.array(frames)
        return frames

    def _get_frames_from_images(self, imgs_dir: str, start_frame: int, end_frame: int) -> torch.Tensor:
        frames = []
        for frame in range(start_frame, end_frame):
            img_path = os.path.join(imgs_dir, f"{frame:05d}.jpg")
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            frames.append(img_tensor)

        return torch.stack(frames)

    def get_n_classes(self) -> int:
        raise NotImplementedError("please implement this method.")

    def get_n_frames(self) -> int:
        raise NotImplementedError("please implement this method.")
