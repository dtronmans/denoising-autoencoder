import os
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from src.config import Config  # Make sure this path is valid


def infer(image_path, model):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(torch.device("cuda"))  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)

    image_output = output.cpu().numpy().squeeze()
    image_output = (image_output * 255).astype(np.uint8)

    if image_output.ndim == 3 and image_output.shape[0] == 3:
        image_output = np.transpose(image_output, (1, 2, 0))

    return image_output


def process_directory(input_dir, output_dir, label, model):
    input_path = os.path.join(input_dir, label)
    output_path = os.path.join(output_dir, label)
    os.makedirs(output_path, exist_ok=True)

    image_filenames = os.listdir(input_path)
    random.shuffle(image_filenames)

    for filename in tqdm(image_filenames, desc=f"Processing {label}"):
        input_image_path = os.path.join(input_path, filename)
        output_image = infer(input_image_path, model)

        output_image_path = os.path.join(output_path, filename)
        cv2.imwrite(output_image_path, output_image)


def main(input_dir, output_dir):
    config = Config(os.path.join("src", "config.json"))
    model = config.architecture
    model.load_state_dict(torch.load("model336_544.pt", map_location=torch.device('cpu')))
    model.eval()

    for label in ["benign", "malignant"]:
        process_directory(input_dir, output_dir, label, model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch infer and save output images")
    parser.add_argument("--input_dir", required=True, help="Path to input directory with benign and malignant subdirs")
    parser.add_argument("--output_dir", required=True, help="Destination directory to save outputs")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
