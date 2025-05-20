import random

from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import cv2
import os
from tqdm import tqdm

from src.config import Config


def infer(image_path, model, show=True):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    output = model(image_tensor)

    image_output = output.detach().cpu().numpy().squeeze()
    image_output = (image_output * 255).astype(np.uint8)

    if image_output.shape[0] == 3:
        image_output = np.transpose(image_output, (1, 2, 0))

    original_image = np.array(image.resize((544, 336)))


    side_by_side = np.hstack((original_image, image_output))

    if show:
        cv2.imshow('Original and Predicted', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_output


if __name__ == "__main__":
    path = "../final_datasets/once_more/mtl_final/images/benign"
    # path = "denoiser_dataset/annotated"
    config = Config(os.path.join("src", "config.json"))
    config.architecture.load_state_dict(torch.load("denoiser_mse.pt", weights_only=True, map_location=torch.device('cpu')))
    config.architecture.eval()
    paths = os.listdir(path)
    random.shuffle(paths)
    for image in tqdm(paths):
        if image.endswith("json"):
            continue
        cleaned_image = infer(os.path.join(path, image), config.architecture, show=True)
        cv2.imwrite(os.path.join(path, image), cleaned_image)
