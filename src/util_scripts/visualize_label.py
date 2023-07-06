import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.golfdb import GolfDB


def main():
    root_dir = "data/golfdb/videos_160"
    annotation_file = "data/golfdb/val_split_1.pkl"
    seq_length = 32
    transform = None
    train = False
    use_label_distribution = False
    use_other_phase_label = False
    sigma_label_distribution = 0.8
    use_middle_phase_label = True
    use_temporal_label_smoothing = False

    dataset = GolfDB(
        root_dir,
        annotation_file,
        seq_length,
        transform,
        train,
        use_label_distribution,
        use_other_phase_label,
        sigma_label_distribution,
        use_middle_phase_label,
        use_temporal_label_smoothing,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, data in enumerate(dataloader):
        label = data["labels"]
        gts = label.max(dim=1)[1].cpu().numpy()[0]
        seq_len = label.shape[1]
        graph = np.zeros((seq_len, 8))
        for j in range(8):
            graph[gts[j], j] = 1

        for j in range(8):
            plt.plot(graph[:, j], label=str(j))
        plt.savefig("label.png")
        plt.close()

        label = label.view(-1, 9)
        # gts = label.max(dim=1)[1]

        for j in range(8):
            plt.axvspan(gts[j], gts[j + 1], color="C{}".format(j), alpha=0.5)
        plt.axvspan(gts[8], seq_len, color="C{}".format(9), alpha=0.5)
        plt.savefig("label2.png")
        plt.close()


if __name__ == "__main__":
    main()
