import os.path as osp
from logging import getLogger
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = getLogger(__name__)


class GolfDB(Dataset):
    def __init__(
        self,
        data_file: str,
        vid_dir: str,
        seq_length: int,
        transform: Optional[transforms.Compose] = None,
        train=True,
    ):
        try:
            self.df = pd.read_pickle(data_file)
        except FileNotFoundError("data file not found.") as e:
            logger.exception(f"{e}")

        self.vid_dir: str = vid_dir
        self.seq_length: int = seq_length
        self.transform: Optional[transforms.Compose] = transform
        self.train: bool = train
        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        a = self.df.loc[idx, :]  # annotation info
        events = a["events"]
        events -= events[
            0
        ]  # now frame #s correspond to frames in preprocessed video clips

        images, flows, labels = [], [], []
        cap = cv2.VideoCapture(
            osp.join(self.vid_dir, "{}.mp4".format(a["id"]))
        )

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {"images": np.asarray(images), "labels": np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels = sample["images"], sample["labels"]
        images = images.transpose((0, 3, 1, 2))
        return {
            "images": torch.from_numpy(images).float().div(255.0),
            "labels": torch.from_numpy(labels).long(),
        }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample["images"], sample["labels"]
        images.sub_(self.mean[None, :, None, None]).div_(
            self.std[None, :, None, None]
        )
        return {"images": images, "labels": labels}


def get_inputs(sample):
    return (
        torch.stack((sample["images"], sample["flows"]), dim=0),
        sample["labels"],
    )


def get_dataset(config):
    return GolfDB(
        config.DATASET.DATA_FILE,
        config.DATASET.VID_DIR,
        config.DATASET.SEQ_LENGTH,
        config.DATASET.TRANSFORM,
        config.DATASET.TRAIN,
    )


if __name__ == "__main__":
    norm = Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )  # ImageNet mean and std (RGB)

    dataset = GolfDB(
        data_file="data/train_split_1.pkl",
        vid_dir="data/videos_160/",
        seq_length=64,
        transform=transforms.Compose([ToTensor(), norm]),
        train=False,
    )

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False
    )

    for i, sample in enumerate(data_loader):
        images, labels = sample["images"], sample["labels"]
        events = np.where(labels.squeeze() < 8)[0]
        print("{} events: {}".format(len(events), events))
