import json

from torchvision.transforms import transforms

from src.architectures import Autoencoder, AutoencoderWithSkipConnections
from src.datasets import UltrasoundDataset


class Config:
    def __init__(self, json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)

        # Dataset
        self.dataset = self.get_nested(config, ['dataset', 'dataset'])
        self.dataset_path = self.get_nested(config, ['dataset', 'dataset_path'])
        self.clean_images_txt = self.get_nested(config, ['dataset', 'clean_images_txt'])

        # Training
        self.architecture = self.get_nested(config, ['training', 'architecture'])
        self.loss_alpha = self.get_nested(config, ['training', 'loss_alpha'])
        self.loss_beta = self.get_nested(config, ['training', 'loss_beta'])
        self.resize_size = self.get_nested(config, ['training', 'resize_size'])
        self.epochs = self.get_nested(config, ['training', 'epochs'])
        self.lr = self.get_nested(config, ['training', 'lr'])
        self.batch_size = self.get_nested(config, ['training', 'batch_size'])
        self.val_split = self.get_nested(config, ['training', 'val_split'])

        # Draw
        self.interactive_mode = self.get_nested(config, ['draw', 'interactive_mode'], default=False)

        self.parse_architecture_dataset()

    def __repr__(self):
        return f"Config({self.__dict__})"

    def get_nested(self, dictionary, keys, default=None):
        for key in keys:
            if isinstance(dictionary, dict):
                dictionary = dictionary.get(key)
            else:
                return default
        return dictionary

    def parse_architecture_dataset(self):
        self.transforms = transforms.Compose([
            transforms.Resize((384, 384)),  # Resize to a fixed size
            transforms.ToTensor(),  # Convert to tensor
        ])

        if self.dataset == "UltrasoundDataset":
            self.dataset = UltrasoundDataset(self.dataset_path, transforms=self.transforms)
        else:
            raise ValueError("Not a correct selected dataset")

        if self.architecture == "Autoencoder":
            self.architecture = Autoencoder()
        elif self.architecture == "AutoencoderWithSkipConnections":
            self.architecture = AutoencoderWithSkipConnections()
