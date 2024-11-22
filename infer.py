from enum import Enum

from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import cv2
import os

from architectures import Autoencoder
from architecture_skip_connections import AutoencoderWithSkipConnections


class Model(Enum):
    NORMAL = Autoencoder()
    SKIPNET = AutoencoderWithSkipConnections()


def infer(image_path, show=True, model=Model.NORMAL):
    if model == Model.NORMAL:
        model = Autoencoder()
    else:
        model = AutoencoderWithSkipConnections()
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()

    # Load and transform the input image
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Get the output from the model
    output = model(image_tensor)

    # Convert the output to a displayable format
    image_output = output.detach().cpu().numpy().squeeze()
    image_output = (image_output * 255).astype(np.uint8)

    # Convert the original image to a NumPy array for side-by-side display
    original_image = np.array(image.resize((256, 256)))

    # Stack the original and predicted images side-by-side
    side_by_side = np.hstack((original_image, image_output))

    if show:
        cv2.imshow('Original and Predicted', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # infer("dataset/annotated/92.JPG", model=Model.SKIPNET)
    path = os.path.join("dataset", "annotated")

    for image in os.listdir(path):
        infer(os.path.join(path, image), model=Model.SKIPNET)
