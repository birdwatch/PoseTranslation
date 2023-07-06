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

from src.datasets.video_dataset import VideoDataset

logger = getLogger(__name__)


class GolfDB(VideoDataset):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        seq_length: int,
        transform: Optional[transforms.Compose] = None,
        train: bool = True,
        use_label_distribution: bool = False,
        use_other_phase_label: bool = False,
        sigma_label_distribution: float = 0.8,
        use_middle_phase_label: bool = False,
        use_temporal_label_smoothing: bool = False,
    ) -> None:
        super().__init__(root_dir, annotation_file, transform)

        self.seq_length = seq_length
        self.train = train
        if use_other_phase_label:
            self.n_classes = 2 * len(self.df.loc[0, "events"]) - 3
        elif use_middle_phase_label:
            self.n_classes = len(self.df.loc[0, "events"]) - 1
        else:
            self.n_classes = len(self.df.loc[0, "events"]) - 1
        self.use_label_distribution = use_label_distribution
        self.use_other_phase_label = use_other_phase_label
        self.sigma_label_distribution = sigma_label_distribution
        self.use_middle_phase_label = use_middle_phase_label
        self.use_temporal_label_smoothing = use_temporal_label_smoothing

        logger.info("seq_length: {}".format(self.seq_length))
        logger.info("train: {}".format(self.train))
        logger.info("n_classes: {}".format(self.n_classes))
        logger.info("use_label_distribution: {}".format(self.use_label_distribution))
        logger.info("use_other_phase_label: {}".format(self.use_other_phase_label))
        logger.info("sigma_label_distribution: {}".format(self.sigma_label_distribution))
        logger.info("use_middle_phase_label: {}".format(self.use_middle_phase_label))
        logger.info("use_temporal_label_smoothing: {}".format(self.use_temporal_label_smoothing))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid_path = os.path.join(self.root_dir, "{}.mp4".format(self.df.loc[idx, "id"]))
        assert os.path.exists(vid_path), "{} does not exist.".format(vid_path)
        phases = self.df.loc[idx, "events"]
        phases -= phases[0]
        phases = phases[1:-1]
        vid_cap = cv2.VideoCapture(vid_path)
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count != 0, "frame_count is 0."
        vid_cap.release()

        if self.train:
            start_frame = np.random.randint(0, frame_count - self.seq_length)
            end_frame = start_frame + self.seq_length
        else:
            start_frame = 0
            end_frame = frame_count
        start_frame = 0
        end_frame = frame_count

        frames = self._get_frames_from_video(vid_path, start_frame, end_frame)
        labels = self._generate_label(phases, frame_count, start_frame, end_frame)

        sample = {
            "images": frames,
            "labels": labels,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _generate_label(
        self,
        phases: np.ndarray,
        frame_count: int,
        start_frame: int,
        end_frame: int,
    ) -> np.ndarray:
        """
        Args:
            phases: phases of the video
            start_frame: start frame of the video
            end_frame: end frame of the video
            use_label_distribution: use label distribution or not
            sigma_label_distribution: sigma of label distribution
            use_label_smoothing: use label smoothing or not
            use_other_phase_label: use label which named the other phase is where
        Returns:
            label: label of the video
        """
        # assert (not use_other_phase_label) * use_label_distribution is False, "not implemented yet."

        # assert (
        #     use_middle_phase_label * use_other_phase_label is False
        # ), "use_middle_phase_label and use_other_phase_label cannot be True at the same time."

        entire_labels = np.zeros([frame_count, self.n_classes])
        other_label = self.n_classes - 1
        current_frame_label = 0

        for i in range(frame_count):
            if i in phases and (self.use_middle_phase_label or self.use_other_phase_label):
                current_frame_label += 1

            if self.use_label_distribution:
                entire_labels[i] = np.exp(
                    -((np.arange(self.n_classes) - current_frame_label) ** 2) / (2 * self.sigma_label_distribution**2)
                )
                entire_labels[i] /= np.sum(entire_labels[i])
            elif self.use_middle_phase_label or self.use_other_phase_label:
                entire_labels[i, current_frame_label] = 1
            else:
                if i in phases:
                    entire_labels[i, current_frame_label] = 1
                else:
                    entire_labels[i, other_label] = 1

            if i in phases:
                if self.use_other_phase_label or (
                    self.use_other_phase_label is False and self.use_middle_phase_label is False
                ):
                    current_frame_label += 1

        if self.use_temporal_label_smoothing:
            for phase in phases:
                if self.use_middle_phase_label:
                    entire_labels[phase] = np.convolve(entire_labels[phase], np.ones(3) / 3, mode="same")

            # if use_label_distribution:
            #     entire_labels[i] = np.exp(
            #         -((np.arange(self.n_classes) - current_frame_label) ** 2) / (2 * sigma_label_distribution**2)
            #     )
            #     entire_labels[i] /= np.sum(entire_labels[i])
            # if use_middle_phase_label:
            #     if i in phases:
            #         current_frame_label += 1
            #         entire_labels[i, current_frame_label] = 1
            #     else:
            #         entire_labels[i, current_frame_label] = 1
            # elif use_other_phase_label:
            #     if i in phases:
            #         current_frame_label += 1
            #         entire_labels[i, current_frame_label] = 1
            #         current_frame_label += 1
            #     else:
            #         entire_labels[i, current_frame_label] = 1
            # else:
            #     if i in phases:
            #         entire_labels[i, current_frame_label] = 1
            #         current_frame_label += 1
            #     else:
            #         entire_labels[i, other_label] = 1

        # for i, phase in enumerate(phases):
        #     entire_labels[phase, 2 * i + 1] = 1

        return entire_labels[start_frame:end_frame]

    def get_n_classes(self) -> int:
        return self.n_classes


def get_inputs(sample: Dict[str, Any]) -> tuple:
    """_summary_

    Parameters
    ----------
    sample : Dict[str, Any]
        _description_

    Returns
    -------
    tuple
        _description_
    """
    return (
        sample["images"],
        sample["labels"],
    )


def get_dataset(config: Dict, split: str, transform) -> GolfDB:
    """
    Args:
        config: configuration for training
        split: train or val
        Returns:
            dataset: GolfDB dataset
    """
    return GolfDB(
        config.DATASET.VID_DIR,
        config.DATASET.TRAIN_FILE if split == "train" else config.DATASET.VAL_FILE,
        config.DATASET.SEQ_LENGTH,
        transform=transform,
        train=True if split == "train" else False,
        use_label_distribution=config.DATASET.USE_LABEL_DISTRIBUTION,
        use_other_phase_label=config.DATASET.USE_OTHER_PHASE_LABEL,
        sigma_label_distribution=config.DATASET.SIGMA_LABEL_DISTRIBUTION,
        use_middle_phase_label=config.DATASET.USE_MIDDLE_PHASE_LABEL_ONLY,
        use_temporal_label_smoothing=config.DATASET.USE_TEMPORAL_LABEL_SMOOTHING,
    )
