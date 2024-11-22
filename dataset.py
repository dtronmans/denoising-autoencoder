import os
from PIL import Image
from torch.utils.data import Dataset

# for now I resort to this https://github.com/cv516Buaa/MMOTU_DS2Net?tab=readme-ov-file


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

