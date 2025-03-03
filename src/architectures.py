import torch
import torch.nn as nn
import torch.optim as optim


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

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  # Upsample (B, 1, 128, 128)
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

        self.encoder_conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(2, 2)

        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(2, 2)

        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
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

        self.decoder_upsample3 = nn.ConvTranspose2d(128 + 64, 1, kernel_size=2, stride=2)  # Skip connection
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
