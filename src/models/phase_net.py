from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_memlab import profile
from torch.autograd import Variable
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.models.tcn import MultiStageTCN, SingleStageTCN


class PhaseNet(nn.Module):
    def __init__(self, phase_num: int, seq_length: int, device: str) -> None:
        super().__init__()

        # hyperparameters
        self.phase_num = phase_num
        self.seq_length = seq_length
        self.device = device

        # model definition
        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.tcn = MultiStageTCN(
            in_channel=1000,
            n_features=64,
            n_classes=self.phase_num,
            n_stages=4,
            n_layers=self.phase_num,
        )
        self.softmax = nn.Softmax(dim=1)

    # @profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size, entire_seq_length, c, h, w = x.size()

            c_in = x.view(batch_size * entire_seq_length, c, h, w)

            with torch.no_grad():
                c_out = self.cnn(c_in)

            tcn_in = c_out.view(batch_size, 1000, entire_seq_length)
            tcn_outs = self.tcn(tcn_in)

            for i, tcn_out in enumerate(tcn_outs):
                tcn_outs[i] = tcn_out.view(batch_size * entire_seq_length, self.phase_num)

            return tcn_outs

        else:
            batch_size, entire_seq_length, c, h, w = x.size()
            cnn_in = x.view(batch_size * entire_seq_length, c, h, w)
            cnn_out = self.cnn(cnn_in)

            tcn_in = cnn_out.view(batch_size, 1000, entire_seq_length)
            tcn_out = self.tcn(tcn_in)
            tcn_out = tcn_out.view(batch_size * entire_seq_length, self.phase_num)

            return tcn_out


def get_model(config: Dict[str, Any], device: str = "cuda") -> nn.Module:
    model = PhaseNet(
        phase_num=config.MODEL.PHASE_NUM,
        seq_length=config.TRAIN.SEQ_LENGTH,
        device=device,
    )

    return model.to(device)
