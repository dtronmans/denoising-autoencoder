import torch
import torch.nn as nn
import torch.optim as optim

from src.unet_parts import Down, DoubleConv, Up, OutConv


class BasicUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BasicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Autoencoder(nn.Module):
    def __init__(self, num_channels):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.latent = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, 16, 16)
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, 16, 16)
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample (B, 128, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample (B, 64, 64, 64)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),  # Upsample (B, 1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x


class AutoencoderWithSkipConnections(nn.Module):
    def __init__(self, num_channels):
        super(AutoencoderWithSkipConnections, self).__init__()

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(2, 2)

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(2, 2)

        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder_relu3 = nn.ReLU()
        self.encoder_pool3 = nn.MaxPool2d(2, 2)

        self.latent_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.latent_relu1 = nn.ReLU()
        self.latent_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.latent_relu2 = nn.ReLU()

        self.decoder_upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_relu1 = nn.ReLU()

        self.decoder_upsample2 = nn.ConvTranspose2d(256 + 128, 64, kernel_size=2, stride=2)  # Skip connection
        self.decoder_relu2 = nn.ReLU()

        self.decoder_upsample3 = nn.ConvTranspose2d(128 + 64, 3, kernel_size=2, stride=2)  # Skip connection
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1 = self.encoder_relu1(self.encoder_conv1(x))
        x = self.encoder_pool1(skip1)

        skip2 = self.encoder_relu2(self.encoder_conv2(x))
        x = self.encoder_pool2(skip2)

        skip3 = self.encoder_relu3(self.encoder_conv3(x))
        x = self.encoder_pool3(skip3)

        x = self.latent_relu1(self.latent_conv1(x))
        x = self.latent_relu2(self.latent_conv2(x))

        x = self.decoder_relu1(self.decoder_upsample1(x))

        x = torch.cat((x, skip3), dim=1)  # Concatenate skip3
        x = self.decoder_relu2(self.decoder_upsample2(x))

        x = torch.cat((x, skip2), dim=1)  # Concatenate skip2
        x = self.decoder_sigmoid(self.decoder_upsample3(x))

        return x


if __name__ == "__main__":
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)
