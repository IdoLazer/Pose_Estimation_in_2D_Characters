import os
from datetime import datetime
from pathlib import Path

import json_tricks as json
import numpy as np
import pandas as pd
import torch
import shutil
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

import project_files.ImageGenerator as ImageGenerator
import project_files.Config as Config
from project_files.Config import config


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


def generate_parameters(num_layers, samples_num, angle_range=None, scaling_range=None, translate_range=None):
    if angle_range is None:
        angles = np.zeros(shape=(samples_num, 1, num_layers))
    else:
        angles = np.random.randint(-angle_range, angle_range, size=samples_num * num_layers). \
            reshape((samples_num, 1, num_layers))
    if scaling_range is None:
        x_scaling = np.ones(shape=(samples_num, 1, num_layers))
        y_scaling = np.ones(shape=(samples_num, 1, num_layers))
    else:
        x_scaling = np.random.uniform(scaling_range[0], scaling_range[1], size=samples_num * num_layers). \
            reshape((samples_num, 1, num_layers))
        y_scaling = np.random.uniform(scaling_range[0], scaling_range[1], size=samples_num * num_layers). \
            reshape((samples_num, 1, num_layers))
    if translate_range is None:
        x_translate = np.zeros(shape=(samples_num, 1, num_layers))
        y_translate = np.zeros(shape=(samples_num, 1, num_layers))
    else:
        x_translate = np.random.randint(translate_range[0], translate_range[1], size=samples_num * num_layers). \
            reshape((samples_num, 1, num_layers))
        y_translate = np.random.randint(translate_range[0], translate_range[1], size=samples_num * num_layers). \
            reshape((samples_num, 1, num_layers))
    parameters = np.concatenate((angles, x_scaling,
                                 y_scaling, x_translate, y_translate), axis=1)
    return parameters


def save_image_batch(images, labels, start_idx, annotations, output_path):
    for idx, (image, label) in enumerate(zip(images, labels)):
        name = f"pose{start_idx + idx}.png"
        label["image"] = name
        annotations.append(label)
        image.save(f"{output_path}images\\{name}")


def forge_new_dataset(samples=1000, num_samples_to_save=1000):
    angle_range = config['dataset']['angle_range']
    scaling_range = config['dataset']['scaling_range']
    translation_range = config['dataset']['translation_range']
    data_path = f"{Path(__file__).resolve().parent.parent}\\data\\{config['dataset']['character']}\\"
    output_path = f"{data_path}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\\"
    num_layers = len(ImageGenerator.char.char_tree_array)

    try:
        os.makedirs(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)
    try:
        os.makedirs(f"{output_path}images\\")
    except OSError:
        print(f"Creation of the directory {output_path}images\\ failed")
    try:
        os.makedirs(f"{output_path}annot\\")
    except OSError:
        print(f"Creation of the directory {output_path}annot\\ failed")
    try:
        conf_file = open(output_path + "\\config.txt", "w")
        conf_file.write(Config.serialize())
        conf_file.close()
    except OSError:
        print("Creation of config file failed")

    train_annotations = []
    test_annotations = []
    train_parameters = generate_parameters(num_layers, int(samples * 0.8), angle_range, scaling_range, translation_range)
    test_parameters = generate_parameters(num_layers, int(samples * 0.2), angle_range, scaling_range, translation_range)
    forged_images = []
    batch_annotations = []
    for index in tqdm(range(len(train_parameters))):
        im_parameters = train_parameters[index]
        idx = np.random.randint(2)
        char = ImageGenerator.char if idx == 0 else ImageGenerator.char_side
        im, annotations = ImageGenerator.create_image(char, im_parameters, draw_skeleton=False,
                                                      print_dict=False, as_image=True)
        forged_images.append(im)
        batch_annotations.append(annotations)
        if index % num_samples_to_save == num_samples_to_save - 1:
            save_image_batch(forged_images, batch_annotations, index - num_samples_to_save + 1, train_annotations,
                             output_path)
            forged_images = []
            batch_annotations = []
    json.dump(train_annotations, f"{output_path}annot\\train.json")

    for index in tqdm(range(len(test_parameters))):
        im_parameters = test_parameters[index]
        im, annotations = ImageGenerator.create_image(ImageGenerator.char, im_parameters, draw_skeleton=False,
                                                      print_dict=False, as_image=True)
        forged_images.append(im)
        batch_annotations.append(annotations)
        if index % num_samples_to_save == num_samples_to_save - 1:
            save_image_batch(forged_images, batch_annotations, index - num_samples_to_save + 1 + int(samples * 0.8),
                             test_annotations, output_path)
            forged_images = []
            batch_annotations = []
    json.dump(test_annotations, f"{output_path}annot\\valid.json")

    for file in os.listdir(f"{data_path}test_images"):
        shutil.copy(f"{data_path}test_images\\{file}", f"{output_path}images")

    shutil.copy(f"{data_path}test_annot\\test.json", f"{output_path}annot")


if __name__ == "__main__":
    forge_new_dataset(samples=100000, num_samples_to_save=1000)

