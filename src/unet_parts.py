import torch
import torchvision
from torch import nn
import torch.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SE_Block(nn.Module):
    def __init__(self, in_out_dims, ratio=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_out_dims, in_out_dims // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_out_dims // ratio, in_out_dims, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_dims, out_dims, intermediate_dims=None):
        super(EncoderBlock, self).__init__()

        if intermediate_dims is None:
            intermediate_dims = out_dims

        self.conv_1 = nn.Conv2d(in_dims, intermediate_dims, kernel_size=3, padding=1, bias=False)
        self.seb_1 = SE_Block(intermediate_dims)
        self.conv_2 = nn.Conv2d(intermediate_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.seb_2 = SE_Block(out_dims)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.seb_1(x)
        x = self.conv_2(x)
        x = self.seb_2(x)

        return x


class LatentBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(LatentBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.se_block_1 = SE_Block(out_dims)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_dims, 128, kernel_size=3, padding=d, dilation=d)
            for d in [2, 4, 8, 16]
        ])
        self.conv1x1 = nn.Conv2d(out_dims + len([2, 4, 8, 16]) * 128, out_dims, kernel_size=1)
        self.conv_2 = nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.se_block_2 = SE_Block(out_dims)
        self.up_conv = nn.ConvTranspose2d(out_dims, out_dims // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.se_block_1(x)
        dilated_features = [conv(x) for conv in self.dilated_convs]
        x = torch.cat([x] + dilated_features, dim=1)
        x = self.conv1x1(x)

        x = self.conv_2(x)
        x = self.se_block_2(x)
        x = self.up_conv(x)

        return x

