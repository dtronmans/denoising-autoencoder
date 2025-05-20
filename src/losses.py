import torch
import torch.nn.functional as F
from torch import nn


def ssim_loss(predicted, target, C1=0.01 ** 2, C2=0.03 ** 2, window_size=11):
    mu_x = F.avg_pool2d(predicted, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    sigma_x = F.avg_pool2d(predicted ** 2, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(predicted * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / (ssim_d + 1e-7)
    return 1 - ssim_map.mean()


def mse_loss(predicted, target):
    return F.mse_loss(predicted, target)


class WeightedLoss(nn.Module):
    def __init__(self, alpha=5.0, beta=1.0):
        super(WeightedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, clean, annotated, predicted):
        arrow_mask = (annotated != clean).float()

        weight_map = self.alpha * arrow_mask + self.beta * (1 - arrow_mask)

        pixel_wise_error = (clean - predicted) ** 2

        weighted_error = weight_map * pixel_wise_error
        loss = weighted_error.sum() / weight_map.sum()

        return loss


def total_loss(clean, target, predicted, lambda_mse=1.0, lambda_ssim=0.00):
    mse = WeightedLoss(alpha=150.0, beta=1.0)
    weighted_loss = mse(clean, target, predicted)
    ssim = ssim_loss(predicted, target)
    return lambda_mse * weighted_loss + lambda_ssim * ssim
