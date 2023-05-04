import torch.nn as nn
from architecture.Shared import kaiming_init


class DecoderMNIST(nn.Module):
    def __init__(self, n_channel=3, dim_h=128, dim_y=64):
        super(DecoderMNIST, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_y = dim_y

        self.dec_lin = nn.Sequential(nn.Linear(self.dim_y, self.dim_h * 8 * 7 * 7))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),  # 32 x 32
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),  # 64 x 64
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, 2),  # 64 x 64
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.dec_lin(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x


class DecoderCeleba(nn.Module):
    def __init__(self, n_channel=3, dim_h=128, dim_y=64):
        super(DecoderCeleba, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_y = dim_y

        self.dec_lin = nn.Sequential(nn.Linear(self.dim_y, self.dim_h * 8 * 8 * 8))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),  # 32 x 32
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),  # 64 x 64
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),  # 64 x 64
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 1),
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.dec_lin(x)
        x = x.view(-1, self.dim_h * 8, 8, 8)
        x = self.deconv(x)
        return x


class DecoderCifar10(nn.Module):
    def __init__(self, n_channel=3, dim_h=128, dim_y=64):
        super(DecoderCifar10, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.dim_y = dim_y

        self.dec_lin = nn.Sequential(nn.Linear(self.dim_y, self.dim_h * 2 * 4 * 4), nn.ReLU(True))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h * 2, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 32 x 32
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 64 x 64
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.dec_lin(x)
        x = x.view(-1, self.dim_h * 2, 4, 4)
        x = self.deconv(x)
        return x