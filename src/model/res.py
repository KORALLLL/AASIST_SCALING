import torch.nn as nn
from torch import Tensor


class Residual_block(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False):
        super().__init__()
        self.first = first
        in_channels, out_channels = nb_filts
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), padding=(1, 1))
        self.act = nn.GELU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 3), padding=(0, 1))
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv_downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = x if self.first else self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity