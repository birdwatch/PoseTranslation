import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    resnet50,
)
from torchvision.models.video import R3D_18_Weights, r3d_18

logger = logging.getLogger(__name__)


class SwingNetV2(nn.Module):
    def __init__(
        self,
        phase_num: int = 8,
        backbone: str = "mobilenet_v2",
        width_mult: int = 1,
        lstm_layers: int = 1,
        lstm_hidden: int = 256,
        bidirectional: bool = True,
        dropout: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # hyperparameters
        self.phase_num = phase_num
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        # model definition
        if backbone == "mobilenet_v2":
            self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            cnn_out_channels = 1000
        elif backbone == "resnet50":
            self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
            cnn_out_channels = 2048
        elif backbone == "r3d_18":
            self.cnn = r3d_18(weights=R3D_18_Weights.DEFAULT)
            cnn_out_channels = 512
        else:
            logger.error("You have to choose one of the following backbones: mobilenet_v2, resnet50, r3d_18")
            raise NotImplementedError
        logger.info(f"Backbone: {backbone}")

        if backbone == "mobilenet_v2":
            self.flow_cnn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            flow_cnn_out_channels = 1000
        elif backbone == "resnet50":
            self.flow_cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
            flow_cnn_out_channels = 2048
        elif backbone == "r3d_18":
            self.flow_cnn = r3d_18(weights=R3D_18_Weights.DEFAULT)
            flow_cnn_out_channels = 512
        else:
            logger.error("You have to choose one of the following backbones: mobilenet_v2, resnet50, r3d_18")
            raise NotImplementedError

        self.fusion_feature = nn.Sequential(
            nn.Conv1d(cnn_out_channels + flow_cnn_out_channels, cnn_out_channels, 1),
            nn.ReLU(inplace=True),
        )
        # self.fusion_feature_masked_label = nn.Conv1d(cnn_out_channels + 1, cnn_out_channels, 1)

        self.rnn = nn.LSTM(
            int(cnn_out_channels * self.width_mult if self.width_mult > 1.0 else cnn_out_channels),
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, self.phase_num)
        else:
            self.lin = nn.Linear(self.lstm_hidden, self.phase_num)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size: int) -> tuple:
        if self.bidirectional:
            return (
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True
                ),
                Variable(
                    torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True
                ),
            )
        else:
            return (
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True
                ),
                Variable(
                    torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device), requires_grad=True
                ),
            )

    def forward(self, x: torch.Tensor, masked_label: torch.Tensor = None) -> torch.Tensor:
        batch_size, timesteps, C, H, W = x.size()
        C = int(C / 2)
        img, flow = x[:, :, :3, :, :], x[:, :, 3:, :, :]
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = img.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)

        # Flow CNN forward
        f_in = flow.view(batch_size * timesteps, C, H, W)
        f_out = self.flow_cnn(f_in)

        # Fusion
        fusion_in = torch.cat((c_out, f_out), dim=1)
        fusion_in = fusion_in.view(batch_size, -1, timesteps)
        fusion_out = self.fusion_feature(fusion_in)
        fusion_out = fusion_out.view(batch_size * timesteps, -1)

        if self.dropout:
            fusion_out = self.drop(fusion_out)

        fusion_out = fusion_out.view(batch_size, timesteps, -1)

        # LSTM forward
        r_out, states = self.rnn(fusion_out, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.phase_num)

        return out


def get_model(config: Dict[str, Any], device: str = "cuda") -> nn.Module:
    model = SwingNetV2(
        phase_num=config.MODEL.PHASE_NUM,
        width_mult=config.MODEL.WIDTH_MULT,
        lstm_layers=config.MODEL.LSTM_LAYERS,
        lstm_hidden=config.MODEL.LSTM_HIDEEN,
        bidirectional=config.MODEL.BIDIRECTIONAL,
        dropout=config.MODEL.DROPOUT,
        device=device,
    )

    return model.to(device)
