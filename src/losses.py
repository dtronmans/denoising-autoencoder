import torch
import torch.nn as nn
import pytorch_msssim


class CombinedLoss(nn.Module):
    def __init__(self, lambda_denoise=1.0, lambda_classify=1.0):
        super(CombinedLoss, self).__init__()
        self.lambda_denoise = lambda_denoise
        self.lambda_classify = lambda_classify
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, denoising_output, target_images, classification_output, target_labels):
        """
        Calculate the combined loss for both denoising and classification tasks.

        Parameters:
        - denoising_output: the output from the denoising part of the network.
        - target_images: the clean images (ground truth for denoising).
        - classification_output: the output logits from the classification part of the network.
        - target_labels: the actual labels of the images (ground truth for classification).

        Returns:
        - total_loss: the weighted sum of denoising and classification losses.
        """
        # Compute the MSE loss for denoising
        denoising_loss = self.mse_loss(denoising_output, target_images)

        # Compute the Cross Entropy loss for classification
        classification_loss = self.cross_entropy_loss(classification_output, target_labels)

        # Combine the losses
        total_loss = self.lambda_denoise * denoising_loss + self.lambda_classify * classification_loss

        return total_loss


class DAELosses(nn.Module):
    def __init__(self, alpha=0.8, loss_type="combined"):
        super(DAELosses, self).__init__()

        self.alpha = alpha
        self.loss_type = loss_type
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=1)

    def mse(self, predicted, target):
        return self.mse_loss(predicted, target)

    def ssim(self, predicted, target):
        return 1 - self.ssim_loss(predicted, target)

    def combined(self, predicted, target):
        mse = self.mse(predicted, target)
        ssim = self.ssim(predicted, target)
        return self.alpha * mse + (1 - self.alpha) * ssim

    def forward(self, predicted, target):
        if self.loss_type == "mse":
            return self.mse(predicted, target)
        elif self.loss_type == "ssim":
            return self.ssim(predicted, target)
        elif self.loss_type == "combined":
            return self.combined(predicted, target)
        else:
            raise ValueError(f"Invalid loss_type '{self.loss_type}'. Choose from 'mse', 'ssim', 'combined'.")


class WeightedLoss(nn.Module):
    def __init__(self, alpha=5.0, beta=1.0):
        super(WeightedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, clean, annotated, predicted):
        arrow_mask = (annotated != clean).float()  # Shape: (batch, 1, height, width)

        weight_map = self.alpha * arrow_mask + self.beta * (1 - arrow_mask)

        pixel_wise_error = (clean - predicted) ** 2  # Shape: (batch, 1, height, width)

        weighted_error = weight_map * pixel_wise_error  # Shape: (batch, 1, height, width)
        loss = weighted_error.sum() / weight_map.sum()

        return loss
