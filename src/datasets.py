import os

from PIL import Image
from torch.utils.data import Dataset


class UltrasoundDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.annotated_dir = os.path.join(dataset_path, "annotated")
        self.clean_dir = os.path.join(dataset_path, "clean")
        self.transforms = transforms

        self.filenames = sorted(os.listdir(self.annotated_dir))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        annotated_path = os.path.join(self.annotated_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        annotated_image = Image.open(annotated_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transforms:
            annotated_image = self.transforms(annotated_image)
            clean_image = self.transforms(clean_image)

        return {"annotated": annotated_image, "clean": clean_image}


if __name__ == "__main__":
    dataset = UltrasoundDataset("dataset")
    image = dataset[5]
    print()
