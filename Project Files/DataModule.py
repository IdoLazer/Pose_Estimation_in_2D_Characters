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
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_pickle(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.training_data, self.test_data = torch.utils.data.random_split(self, [int(len(self) * 0.8),
                                                                           len(self) - int(len(self) * 0.8)])

    def __len__(self):
        return self.img_labels.shape[1] - 1

    def __getitem__(self, idx):
        name = self.img_labels.columns[idx]
        img_path = os.path.join(self.img_dir, name)
        image = read_image(img_path, mode=ImageReadMode.RGB_ALPHA).double()
        image = (image - 127.5) / 127.5
        label = torch.tensor(self.img_labels[name].tolist()).double()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_train_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(self.training_data, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle)


def get_dataset():
    output_path = f"{config['dirs']['source_dir']}\\Data\\{config['dataset']['character']}\\{config['dataset']['name']}"
    return ImageDataset(f"{output_path}\\annotations.pkl", output_path)


def forge_new_dataset(iterations=1):
    samples = config['dataset']['samples_num']
    angle_range = config['dataset']['angle_range']
    output_path = f"{config['dirs']['source_dir']}Data\\{config['dataset']['character']}\\" \
                  f"samples={samples * iterations} {angle_range=}"
    try:
        os.makedirs(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)
    annotations = dict()
    for iteration in range(iterations):
        images, labels = ImageGenerator.load_data(samples_num=samples,
                                                  angle_range=angle_range)
        for idx, (image, label) in enumerate(zip(images, labels)):
            name = f"pose{iteration*config['dataset']['samples_num'] + idx}.png"
            annotations[name] = label
            image.save(f"{output_path}\\{name}")
    df = pd.DataFrame.from_dict(annotations)
    df.to_pickle(f"{output_path}\\annotations.pkl")


if __name__ == "__main__":
    forge_new_dataset(50)

