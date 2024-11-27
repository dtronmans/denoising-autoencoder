from enum import Enum

from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import cv2
import os

from architectures import Autoencoder, AutoencoderWithSkipConnections
from config import Config


def infer(image_path, show=True):
    config = Config("config.json")
    config.architecture.load_state_dict(torch.load("model3.pt", weights_only=True))
    config.architecture.eval()

    # Load and transform the input image
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((config.resize_size, config.resize_size)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Get the output from the model
    output = config.architecture(image_tensor)

    # Convert the output to a displayable format
    image_output = output.detach().cpu().numpy().squeeze()
    image_output = (image_output * 255).astype(np.uint8)

    # Convert the original image to a NumPy array for side-by-side display
    original_image = np.array(image.resize((config.resize_size, config.resize_size)))

    # Stack the original and predicted images side-by-side
    side_by_side = np.hstack((original_image, image_output))

    if show:
        cv2.imshow('Original and Predicted', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_output


if __name__ == "__main__":
    # infer("dataset/all/92.JPG", model=Model.SKIPNET)
    path = os.path.join("rdg_set", "all")

    for image in os.listdir(path):
        cleaned_image = infer(os.path.join(path, image), show=True)
        # cv2.imwrite(os.path.join("complete_clean", image), cleaned_image)
