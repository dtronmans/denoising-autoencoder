# autoencoder + classification architecture

import torch.nn as nn
import torch.optim as optim


class AugmentedAutoencoder(nn.Module):
    def __init__(self, num_classes):
        super(AugmentedAutoencoder, self).__init__()

        self.latent = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, H, W)
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, H, W)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample (B, 128, H*2, W*2)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample (B, 64, H*4, W*4)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  # Upsample (B, 1, H*8, W*8)
            nn.Sigmoid()
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),  # Assuming input image is 128x128; adjust if different
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)

        latent = self.latent(x)

        reconstruction = self.decoder(latent)

        classification_output = self.classification_head(latent)

        return reconstruction, classification_output
