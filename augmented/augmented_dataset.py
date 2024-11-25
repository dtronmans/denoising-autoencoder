import os
from PIL import Image
from torch.utils.data import Dataset


class AugmentedUltrasoundDataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.annotated_dir = os.path.join(dataset_path, "annotated")
        self.clean_benign_dir = os.path.join(dataset_path, "clean", "benign")
        self.clean_malignant_dir = os.path.join(dataset_path, "clean", "malignant")
        self.transforms = transforms

        if not os.path.isdir(self.annotated_dir) or not (
                os.path.isdir(self.clean_benign_dir) and os.path.isdir(self.clean_malignant_dir)):
            raise FileNotFoundError(
                "The dataset directory structure is incorrect. Expected 'annotated', 'clean/benign', "
                "and 'clean/malignant' subdirectories.")

        self.filenames = sorted(os.listdir(self.annotated_dir))
        self.file_labels = []

        # Check and record the existence of each file in either benign or malignant subfolders
        for file in self.filenames:
            benign_path = os.path.join(self.clean_benign_dir, file)
            malignant_path = os.path.join(self.clean_malignant_dir, file)
            if os.path.isfile(benign_path):
                self.file_labels.append((file, 'benign'))
            elif os.path.isfile(malignant_path):
                self.file_labels.append((file, 'malignant'))
            else:
                raise FileNotFoundError(
                    f"Missing corresponding clean file for {file} in 'clean/benign' or 'clean/malignant' directory.")

    def __len__(self):
        return len(self.file_labels)

    def __getitem__(self, idx):
        filename, label = self.file_labels[idx]
        annotated_path = os.path.join(self.annotated_dir, filename)
        clean_path = os.path.join(self.clean_benign_dir if label == 'benign' else self.clean_malignant_dir, filename)

        annotated_image = Image.open(annotated_path).convert("L")
        clean_image = Image.open(clean_path).convert("L")

        if self.transforms:
            annotated_image = self.transforms(annotated_image)
            clean_image = self.transforms(clean_image)

        return {"all": annotated_image, "clean": clean_image, "label": label}


if __name__ == "__main__":
    dataset = AugmentedUltrasoundDataset("../augmented_dataset")
    print(dataset[0])
