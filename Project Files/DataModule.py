import os
from datetime import datetime

import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import ImageGenerator
from Config import config


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, device=None):
        self.img_labels = pd.read_pickle(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.training_data, self.test_data = torch.utils.data.random_split(self, [int(len(self) * 0.8),
                                                                           len(self) - int(len(self) * 0.8)])
        self.device = device

    def __len__(self):
        return self.img_labels.shape[1] - 1

    def __getitem__(self, idx):
        name = self.img_labels.columns[idx]
        img_path = os.path.join(self.img_dir, name)
        image = read_image(img_path, mode=ImageReadMode.RGB_ALPHA).double()
        label = torch.tensor(self.img_labels[name].tolist()).double()
        if self.device:
            image = image.to(self.device)
            label = label.to(self.device)
        image = (image - 127.5) / 127.5
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_train_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(self.training_data, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle)


def get_dataset(transform=None, device=None):
    output_path = f"{config['dirs']['source_dir']}\\Data\\{config['dataset']['character']}\\{config['dataset']['name']}"
    return ImageDataset(f"{output_path}\\annotations.pkl", output_path, transform=transform, device=device)


def forge_new_dataset(samples=1000):
    iterations = int(samples / 1000)
    angle_range = config['dataset']['angle_range']
    output_path = f"{config['dirs']['source_dir']}Data\\{config['dataset']['character']}\\" \
                  f"{samples=} {angle_range=}"
    try:
        os.makedirs(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)
    annotations = dict()
    for iteration in range(iterations):
        images, labels = ImageGenerator.load_data(samples_num=1000,
                                                  angle_range=angle_range)
        for idx, (image, label) in enumerate(zip(images, labels)):
            name = f"pose{iteration * 1000 + idx}.png"
            annotations[name] = label
            image.save(f"{output_path}\\{name}")
    df = pd.DataFrame.from_dict(annotations)
    df.to_pickle(f"{output_path}\\annotations.pkl")


if __name__ == "__main__":
    forge_new_dataset(samples=100000)

