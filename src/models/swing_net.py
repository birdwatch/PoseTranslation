import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import mobilenet_v2, resnet18, resnet34, resnet50


class SwingNet(nn.Module):
    def __init__(
        self,
        phase=8,
        width_mult=1,
        lstm_layers=1,
        lstm_hidden=256,
        dropout=True,
    ) -> None:
        self.phase = phase
        self.cnn = mobilenet_v2(pretrained=True)
        self.rnn = nn.LSTM(
            1280,
            256,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lin = nn.Linear(2 * 256, 9)
        self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        return (
            Variable(
                torch.zeros(2, batch_size, 256).cuda(),
                requires_grad=True,
            ),
            Variable(
                torch.zeros(2, batch_size, 256).cuda(),
                requires_grad=True,
            ),
        )

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, self.phase)

        return out
