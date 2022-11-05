import os
from datetime import datetime
from pathlib import Path

import json_tricks as json
import numpy as np
import pandas as pd
import torch
import shutil
import matplotlib.pyplot as plt
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


def save_image_batch(images, labels, start_idx, annotations, output_path, annotations_type):
    for idx, (image, label) in enumerate(zip(images, labels)):
        name = f"pose{start_idx + idx}_{annotations_type}.png"
        label["image"] = name
        annotations.append(label)
        image.save(f"{output_path}images\\{name}")


def forge_new_dataset(samples=1000, num_samples_to_save=1000, name=None):
    angle_range = config['dataset']['angle_range']
    scaling_range = config['dataset']['scaling_range']
    translation_range = config['dataset']['translation_range']
    data_path = f"{Path(__file__).resolve().parent.parent}\\data\\{config['dataset']['character']}\\"
    output_path = f"{data_path}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{name}\\"
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

    train_parameters = generate_parameters(num_layers, int(samples * 0.8), angle_range, scaling_range, translation_range)
    test_parameters = generate_parameters(num_layers, int(samples * 0.2), angle_range, scaling_range, translation_range)

    generate_images_from_parameters(train_parameters, 'train', output_path, num_samples_to_save)
    generate_images_from_parameters(test_parameters, 'valid', output_path, num_samples_to_save)

    for file in os.listdir(f"{data_path}test_images"):
        shutil.copy(f"{data_path}test_images\\{file}", f"{output_path}images")

    shutil.copy(f"{data_path}test_annot\\test.json", f"{output_path}annot")


def generate_images_from_parameters(parameters, annotations_type, output_path, num_samples_to_save):
    annotations = []
    forged_images = []
    batch_annotations = []
    for index in tqdm(range(len(parameters))):
        im_parameters = parameters[index]
        idx = np.random.randint(config['dataset']['num_frames'])
        if idx == 0:
            char = ImageGenerator.char
        elif idx == 1:
            char = ImageGenerator.char_side
        else:
            char = ImageGenerator.char_back
        im, im_annotations = ImageGenerator.create_image(char, im_parameters, draw_skeleton=False,
                                                      print_dict=False, as_image=True,
                                                      random_order=config['dataset']['augmentations'],
                                                      random_generation=config['dataset']['augmentations'])
        forged_images.append(im)
        batch_annotations.append(im_annotations)
        if index % num_samples_to_save == num_samples_to_save - 1:
            save_image_batch(forged_images, batch_annotations, index - num_samples_to_save + 1, annotations,
                             output_path, annotations_type)
            forged_images = []
            batch_annotations = []
    json.dump(annotations, f"{output_path}annot\\{annotations_type}.json")


if __name__ == "__main__":

    experiments = [
        {'num_frames': 1, 'angle_range': 80, 'size': 50000, 'augmentations': False},
        {'num_frames': 2, 'angle_range': 80, 'size': 50000, 'augmentations': False},
        {'num_frames': 3, 'angle_range': 80, 'size': 50000, 'augmentations': False},
        {'num_frames': 3, 'angle_range': 80, 'size': 50000, 'augmentations': True},
        {'num_frames': 3, 'angle_range': 80, 'size': 10000, 'augmentations': True},
        {'num_frames': 3, 'angle_range': 80, 'size': 100000, 'augmentations': True},
        {'num_frames': 3, 'angle_range': 120, 'size': 50000, 'augmentations': True},
        {'num_frames': 3, 'angle_range': 40, 'size': 50000, 'augmentations': True},
        {'num_frames': 2, 'angle_range': 80, 'size': 50000, 'augmentations': True},
        {'num_frames': 2, 'angle_range': 120, 'size': 50000, 'augmentations': False},
        {'num_frames': 2, 'angle_range': 120, 'size': 50000, 'augmentations': True},
        {'num_frames': 3, 'angle_range': 80, 'size': 100000, 'augmentations': False},
    ]
    for experiment in experiments:
        num_frames = experiment['num_frames']
        angle_range = experiment['angle_range']
        augmentations = experiment['augmentations']
        size = experiment['size']
        config['dataset']['num_frames'] = num_frames
        config['dataset']['angle_range'] = angle_range
        config['dataset']['augmentations'] = augmentations
        forge_new_dataset(samples=size, num_samples_to_save=10,
                          name=f"{size=}-{angle_range=}-{augmentations=}-{num_frames=}")

