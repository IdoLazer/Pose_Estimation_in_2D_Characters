import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

import ImageGenerator
from Config import config


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_dataset(iterations=1):
    for i in range(iterations):
        data = ImageGenerator.load_data(batch_size=config['dataset']['batch_size'],
                                        samples_num=config['dataset']['samples_num'],
                                        angle_range=config['dataset']['angle_range'])
