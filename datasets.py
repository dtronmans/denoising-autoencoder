import os

import cv2
from PIL import Image
from torch.utils.data import Dataset

from draw_utils import DrawUtils


# for now I resort to this https://github.com/cv516Buaa/MMOTU_DS2Net?tab=readme-ov-file
# extra idea: only add the arrow with a random chance, so it learns to identity map images
# that do not have arrows

class UltrasoundDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.annotated_dir = os.path.join(dataset_path, "annotated")
        self.clean_dir = os.path.join(dataset_path, "clean")
        self.transforms = transforms

        if not os.path.isdir(self.annotated_dir) or not os.path.isdir(self.clean_dir):
            raise FileNotFoundError(
                "The dataset directory structure is incorrect. Expected 'annotated' and 'clean' subdirectories.")

        self.filenames = sorted(os.listdir(self.annotated_dir))

        for file in self.filenames:
            clean_path = os.path.join(self.clean_dir, file)
            if not os.path.isfile(clean_path):
                raise FileNotFoundError(f"Missing corresponding clean file for {file} in 'clean' directory.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        annotated_path = os.path.join(self.annotated_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        annotated_image = Image.open(annotated_path).convert("L")
        clean_image = Image.open(clean_path).convert("L")

        if self.transforms:
            annotated_image = self.transforms(annotated_image)
            clean_image = self.transforms(clean_image)

        return {"annotated": annotated_image, "clean": clean_image}


class UltrasoundDatasetInPlaceArrows(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.annotated_dir = os.path.join(dataset_path, "annotated")
        self.transforms = transforms

        if not os.path.isdir(self.annotated_dir):
            raise FileNotFoundError(
                "The dataset directory structure is incorrect. Expected 'annotated' subdirectories.")

        self.filenames = sorted(os.listdir(self.annotated_dir))

    def __len__(self):
        return len(self.filenames)

    # note: the arrow will be drawn in a different location during the training loop
    # the only fix is to first draw the arrows then save the images instead of on-the-fly
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        annotated_path = os.path.join(self.annotated_dir, filename)

        annotated_image = cv2.imread(annotated_path)
        clean_image = DrawUtils.draw_arrows(annotated_image)

        annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).convert("L")
        clean_image = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)).convert("L")

        if self.transforms:
            annotated_image = self.transforms(annotated_image)
            clean_image = self.transforms(clean_image)

        return {"annotated": annotated_image, "clean": clean_image}


class RdGUltrasoundDataset(Dataset):

    def __init__(self, dataset_path, text_path, transforms=None):
        self.annotated_dir = os.path.join(dataset_path, "annotated")
        self.transforms = transforms

        if not os.path.isdir(self.annotated_dir):
            raise FileNotFoundError(
                "The dataset directory structure is incorrect. Expected 'annotated' subdirectories.")

        with open(text_path, 'r') as file:
            valid_files = [line.strip() for line in file]

        valid_files = [f"RdGG_{filename}.png" for filename in valid_files]

        self.filenames = [file for file in sorted(os.listdir(self.annotated_dir)) if file in valid_files]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        annotated_path = os.path.join(self.annotated_dir, filename)

        annotated_image = cv2.imread(annotated_path)
        clean_image = DrawUtils.draw_arrows(annotated_image)  # Assuming a method to draw arrows

        annotated_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).convert("L")
        clean_image = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)).convert("L")

        if self.transforms:
            annotated_image = self.transforms(annotated_image)
            clean_image = self.transforms(clean_image)

        return {"annotated": annotated_image, "clean": clean_image}



if __name__ == "__main__":
    dataset = UltrasoundDataset("dataset")
    image = dataset[5]
    print()
