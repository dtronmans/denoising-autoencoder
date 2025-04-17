import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.unet_parts import EncoderBlock, LatentBlock, SE_Block


class JungUNet(nn.Module):

    def __init__(self, num_channels=3):
        super(JungUNet, self).__init__()

        # first block
        self.encoder_1 = EncoderBlock(num_channels, 128)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2)

        # second block
        self.encoder_2 = EncoderBlock(128, 256)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2)

        # third block
        self.encoder_3 = EncoderBlock(256, 512)
        self.avg_pool_3 = nn.AvgPool2d(kernel_size=2)

        self.latent_block = LatentBlock(512, 512)

        self.encoder_4 = EncoderBlock(512 + 256, 256, 256)
        self.up_conv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.encoder_5 = EncoderBlock(256 + 128, 128, 128)
        self.up_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.encoder_6 = EncoderBlock(128 + 64, 64, 64)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        encoder_1_output = self.encoder_1(x)
        x = self.avg_pool_1(encoder_1_output)

        encoder_2_output = self.encoder_2(x)
        x = self.avg_pool_2(encoder_2_output)

        encoder_3_output = self.encoder_3(x)
        x = self.avg_pool_3(encoder_3_output)

        x = self.latent_block(x)

        x = self.up_conv_1(self.encoder_4(torch.cat([encoder_3_output, x], dim=1)))
        x = self.up_conv_2(self.encoder_5(torch.cat([encoder_2_output, x], dim=1)))
        x = self.last_conv(self.encoder_6(torch.cat([encoder_1_output, x], dim=1)))

        return torch.sigmoid(x)





if __name__ == "__main__":
    image_path = "denoiser_dataset/clean/LUM00054_2.tif"
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 672)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    print(image_tensor.shape)
    model = JungUNet()
    output = model(image_tensor)
    print(output.shape)
