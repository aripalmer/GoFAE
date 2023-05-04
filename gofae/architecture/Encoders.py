import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.Shared import kaiming_init


class EncoderP2(nn.Module):
    def __init__(self, dim_v=256, dim_y=64):
        super(EncoderP2, self).__init__()

        self.dim_v = dim_v
        self.dim_y = dim_y
        self.fc = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.dim_y, self.dim_v)).t())

    def forward(self, x):
        x = F.linear(x, self.fc.t())
        return x


class EncoderMNISTP1(nn.Module):
    def __init__(self, n_channel=1, dim_h=128, dim_v=256, dim_y=64):
        super(EncoderMNISTP1, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_v = dim_v
        self.dim_y = dim_y
        self.img_size = 28

        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h*2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h*4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h*8),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(self.dim_h*(2**3),self.dim_v),
            nn.BatchNorm1d(self.dim_v),
            nn.ReLU(True),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderCelebaP1(nn.Module):
    def __init__(self, n_channel=3, dim_h=128, dim_v=256, dim_y=64):
        super(EncoderCelebaP1, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_v = dim_v
        self.dim_y = dim_y
        self.img_size = 64

        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(self.dim_h * (2 ** 3) * 4 * 4, self.dim_v),
            nn.BatchNorm1d(self.dim_v),
            nn.ReLU(True),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderCifar10P1(nn.Module):
    def __init__(self, n_channel=3, dim_h=128, dim_v=256, dim_y=64):
        super(EncoderCifar10P1, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_v = dim_v
        self.dim_y = dim_y
        self.img_size = 32

        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(self.dim_h * (2 ** 3) * 2 * 2, self.dim_v),
            nn.BatchNorm1d(self.dim_v),
            nn.ReLU(True),
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.conv(x)
        return x
