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


def sobel_filter():
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    return sobel_x, sobel_y


def edge_loss(predicted, target):
    sobel_x, sobel_y = sobel_filter()
    sobel_x = sobel_x.to(predicted.device)
    sobel_y = sobel_y.to(predicted.device)

    grad_pred_x = F.conv2d(predicted, sobel_x, padding=1)
    grad_pred_y = F.conv2d(predicted, sobel_y, padding=1)
    grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)

    grad_target_x = F.conv2d(target, sobel_x, padding=1)
    grad_target_y = F.conv2d(target, sobel_y, padding=1)
    grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + 1e-6)

    return F.l1_loss(grad_pred, grad_target)


def l1_loss(predicted, target):
    return F.l1_loss(predicted, target)


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


def total_loss(clean, predicted, target, lambda_mse=1.0, lambda_ssim=0.2):
    mse = WeightedLoss()
    weighted_loss = mse(clean, predicted, target)
    ssim = ssim_loss(predicted, target)
    return lambda_mse * weighted_loss + lambda_ssim * ssim
