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

from src.models.ASFormer import ASFormer

logger = logging.getLogger(__name__)


class SwingNetV3(nn.Module):
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

        self.asformer = ASFormer(
            num_decoders=1,
            num_layers=3,
            r1=4,
            r2=4,
            num_f_maps=64,
            input_dim=cnn_out_channels,
            num_classes=phase_num,
            channel_masking_rate=0.25,
            device=device,
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)
        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        if self.dropout:
            c_out = self.drop(c_out)

        r_in = c_out.view(batch_size, -1, timesteps)

        # ASFormer forward
        out = self.asformer(r_in, mask)

        return out

    def make_mask(self, label: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(label)
        mask[label == 0] = 0
        return mask


class Encoder(nn.Module):
    def __init__(self, features: int, hidden_dim: int, device: str = "cuda") -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(features, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.lstm(x, hidden)
        return hidden

    def init_hidden(self, batch_size: int) -> tuple:
        return (
            Variable(torch.zeros(2, batch_size, self.hidden_dim).to(self.device), requires_grad=True),
            Variable(torch.zeros(2, batch_size, self.hidden_dim).to(self.device), requires_grad=True),
        )


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: int, device: str = "cuda") -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(1, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        return output, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: int, device: str = "cuda") -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(1, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)

        t_output = output.transpose(1, 2)
        s = torch.bmm(t_output, output)
        attention_weights = torch.softmax(s, dim=2)
        output = torch.bmm(output, attention_weights)
        output = output.transpose(1, 2)
        return output, hidden


def get_model(config: Dict[str, Any], device: str = "cuda") -> nn.Module:
    model = SwingNetV3(
        phase_num=config.MODEL.PHASE_NUM,
        width_mult=config.MODEL.WIDTH_MULT,
        lstm_layers=config.MODEL.LSTM_LAYERS,
        lstm_hidden=config.MODEL.LSTM_HIDEEN,
        bidirectional=config.MODEL.BIDIRECTIONAL,
        dropout=config.MODEL.DROPOUT,
        device=device,
    )

    return model.to(device)
