import json

from torchvision.transforms import transforms

from architectures import Autoencoder, AutoencoderWithSkipConnections
from datasets import UltrasoundDataset, UltrasoundDatasetInPlaceArrows, RdGUltrasoundDataset


class Config:
    def __init__(self, json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)
        self.architecture = config.get('architecture')
        self.dataset = config.get('dataset')
        self.dataset_path = config.get('dataset_path')
        self.clean_images_txt = config.get('clean_images_txt')
        self.loss_alpha = config.get('loss_alpha')
        self.loss_beta = config.get('loss_beta')
        self.resize_size = config.get('resize_size')
        self.epochs = config.get('epochs')
        self.lr = config.get('lr')
        self.batch_size = config.get('batch_size')
        self.parse_architecture_dataset()

    def __repr__(self):
        return f"Config({self.__dict__})"

    def parse_architecture_dataset(self):
        self.transforms = transforms.Compose([
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.ToTensor()
        ])

        if self.dataset == "UltrasoundDataset":
            self.dataset = UltrasoundDataset(self.dataset_path, transforms=self.transforms)
        elif self.dataset == "UltrasoundDatasetInPlaceArrows":
            self.dataset = UltrasoundDatasetInPlaceArrows(self.dataset_path, transforms=self.transforms)
        elif self.dataset == "RdGUltrasoundDataset":
            self.dataset = RdGUltrasoundDataset(self.dataset_path, self.clean_images_txt, transforms=self.transforms)
        else:
            raise ValueError("Not a correct selected dataset")

        if self.architecture == "Autoencoder":
            self.architecture = Autoencoder()
        elif self.architecture == "AutoencoderWithSkipConnections":
            self.architecture = AutoencoderWithSkipConnections()
