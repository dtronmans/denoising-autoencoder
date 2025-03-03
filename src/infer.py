from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import cv2
import os
from tqdm import tqdm

from src.config import Config


def infer(image_path, show=True):
    config = Config(os.path.join("src", "config.json"))
    config.architecture.load_state_dict(torch.load("model384_672.pt", weights_only=True, map_location=torch.device('cpu')))
    config.architecture.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 672)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    output = config.architecture(image_tensor)

    image_output = output.detach().cpu().numpy().squeeze()
    image_output = (image_output * 255).astype(np.uint8)

    if image_output.shape[0] == 3:
        image_output = np.transpose(image_output, (1, 2, 0))

    original_image = np.array(image.resize((672, 384)))

    side_by_side = np.hstack((original_image, image_output))

    if show:
        cv2.imshow('Original and Predicted', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_output


if __name__ == "__main__":
    path = os.path.join("train_set", "all")

    for image in tqdm(os.listdir(path)):
        cleaned_image = infer(os.path.join(path, image), show=True)
        # cv2.imwrite(os.path.join("all_datasets/LUMC_util_png_inferred/malignant", image), cleaned_image)
