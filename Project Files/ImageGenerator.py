import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

source_path = Path(__file__).resolve()
source_dir = source_path.parent
PATH = str(source_dir) + '\\'


class Vector2D:

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector2D(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector2D(x, y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __floordiv__(self, other):
        self.x /= other
        self.y /= other
        return self

    def __str__(self):
        return "({0},{1})".format(self.x, self.y)


images = dict()

# sample_params = [5, 14, -29, -16, 22, -10, -22, 14, -21, -26, -21]
sample_params = [5, 14, -29, -16, 22, -10, -22, 14, -21, -26, -21]


class Character:
    image_size: int  # height in pixels of character image (each layer should be the same size and have height = width)
    path: str  # path to character layers directory
    char_tree: dict  # a dictionary representing the hierarchical structure of the layers
    layers_info: dict  # a dict with relevant info of each layer
    char_tree_array: list  # the char tree as an array - ordered in a BFS
    sample_params: list  # starting pose for character creation, ordered respective to the char_tree_array
    parents: list  # each part parent's index, ordered respective to the char_tree_array
    drawing_order: list  # the order layers should be drawn (by layer name)
    canonical_bias_dict: dict  # a dict with default values for the network to use as bias in the final fc layer

    def __init__(self, path=None):
        if path is not None:
            char = json.load(open(path, 'r'))
            self.image_size = char['image_size']
            self.path = char['path']
            self.char_tree = char['char_tree']
            self.layers_info = char['layers_info']
            for key in self.layers_info:
                item = self.layers_info[key]
                item['displacement'] = Vector2D(item['displacement'][0], item['displacement'][1])
            self.char_tree_array = char['char_tree_array']
            self.sample_params = char['sample_params']
            self.drawing_order = char['drawing_order']
            self.parents = char['parents']
            self.canonical_bias_dict = char['canonical_bias_dict']


    @staticmethod
    def create_default_config_file(path, root_name, char_tree, drawing_order):
        char = Character()

        # find image size
        char.image_size = Image.open(path + root_name + '.png').size[0]
        char.path = path
        char.char_tree = char_tree
        char.layers_info = dict()
        char_tree_array = ["Root"]
        index = 0
        parents = [None]
        while index < len(char_tree_array):
            part = char_tree_array[index]
            char.layers_info[part] = \
                {
                    "displacement": [0, 0],
                    "name": part,
                    "path": path + (root_name if part == "Root" else part) + '.png'
                }
            if part in char_tree:
                char_tree_array += char_tree[part]
                parents += [index] * len(char_tree[part])
            index += 1
        char.char_tree_array = char_tree_array
        char.drawing_order = drawing_order
        char.parents = parents
        char.sample_params = [0] * len(char.char_tree_array)
        json.dump(char, open(char.path + 'Config.txt', 'w'), default=lambda o: o.__dict__,
                  sort_keys=True, indent=4)


class BodyPart:
    def __init__(self, parent, name, path, dist_from_parent, inner_rotation):
        self.parent = parent
        self.name = name
        if name not in images and path is not None:
            self.im = Image.open(path)
            images[name] = self.im
        elif path is not None:
            self.im = images[name]
        joint_rotation = math.radians(0)
        center = Vector2D()
        if parent is not None:
            parent.__add_child(self)
            joint_rotation = parent.rotation
            center = parent.position
        # create_affine_transform(math.radians(inner_rotation), Vector2D(), dist_from_parent,
        #                         name, True, self.im.size[0])  # TODO: This is just to generate initial transformations
        self.position = rotate(center, center + dist_from_parent, -joint_rotation)
        self.center = center
        self.rotation = joint_rotation + math.radians(inner_rotation)
        self.children = []

    def __add_child(self, body_part):
        self.children.append(body_part)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin.x, origin.y
    px, py = point.x, point.y

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return Vector2D(qx, qy)


def translate_points(points, displacement):
    return [points[i] + displacement for i in range(len(points))]


def create_affine_transform(angle, center, displacement, name, print_dict=False, image_size=128):
    dx, dy = (displacement.x, -displacement.y) if print_dict else \
        (displacement.x, -displacement.y)
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y = (0, 0) if print_dict else (center.x, center.y)
    nx, ny = x + dx, y + dy
    a = cos / 1
    b = sin / 1
    c = x - a * nx - b * ny
    d = -sin / 1
    e = cos / 1
    f = y - d * nx - e * ny
    if print_dict:
        bias = [a, b, c, d, e, f]
        print('\"' + name + '\" : ' + str(bias) + ',')
    return np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1]])


def traverse_tree(cur, size, layers, draw_skeleton, skeleton_draw, transform=True, print_dict=False):
    image_center = Vector2D(size / 2, size / 2)
    im = cur.im
    if transform:
        im = im.transform((size, size),
                          Image.AFFINE,
                          data=create_affine_transform(cur.rotation, image_center, cur.position, cur.name, print_dict).
                          flatten()[:6],
                          resample=Image.BILINEAR)
    for child in cur.children:
        if draw_skeleton:
            line = translate_points([cur.position, child.position], image_center)
            skeleton_draw.ellipse(
                xy=(line[1].x - 1, size - (line[1].y + 1), line[1].x + 1, size - (line[1].y - 1),),
                fill="red")
            skeleton_draw.line([(line[0].x, size - line[0].y),
                                (line[1].x, size - line[1].y)], fill="yellow")
        traverse_tree(child, size, layers, draw_skeleton, skeleton_draw, transform, print_dict)
    if not cur.children and draw_skeleton:
        if cur.name == 'Head':
            line = translate_points([cur.position, rotate(cur.position, cur.position + Vector2D(0, 15), -cur.rotation)],
                                    image_center)
        else:
            line = translate_points(
                [cur.position, rotate(cur.position, cur.position + Vector2D(0, -15), -cur.rotation)], image_center)
        skeleton_draw.line([(line[0].x, size - line[0].y),
                            (line[1].x, size - line[1].y)], fill="yellow")
    layers[cur.name] = im


def generate_layers(character, angles, draw_skeleton=False, as_tensor=False, transform=True, print_dict=False):
    origin = create_body_hierarchy(angles, character)
    layers = {}
    skeleton = Image.new('RGBA', (character.image_size, character.image_size))
    skeleton_draw = ImageDraw.Draw(skeleton)
    traverse_tree(origin, character.image_size, layers, draw_skeleton, skeleton_draw, transform, print_dict)
    if not as_tensor:
        layers['Skeleton'] = skeleton
        return layers
    layers_list = []
    for part in character.char_tree_array:
        im = Image.new("RGBA", (character.image_size, character.image_size))
        alpha = ImageOps.invert(layers[part].split()[-1])
        layer = Image.composite(im, layers[part], alpha)
        # layer = layer.convert("RGB")
        layer = np.array(layer)
        layer = (layer - 127.5) / 127.5
        layers_list.append(layer)
    return torch.tensor(np.array(layers_list, dtype='float64'))


def create_image(character, angles, draw_skeleton=False, omit_layers=False, print_dict=False, as_image=False):
    drawing_order = character.drawing_order + ['Skeleton']
    layers = generate_layers(character, angles, draw_skeleton, print_dict=print_dict)
    im_size = character.image_size
    im = Image.new("RGBA", (im_size, im_size))
    for part in drawing_order:
        if omit_layers and part != 'Right Arm':
            continue
        alpha = ImageOps.invert(layers[part].split()[-1])
        im = Image.composite(im, layers[part], alpha)
    return im.convert("RGB") if as_image else np.array(im).astype('uint8')


def create_body_hierarchy(parameters, character):
    for i in range(len(parameters)):
        parameters[i] += character.sample_params[i]
    # parameters = sample_params  # TODO: Always comment out when starting
    parts_list = []
    for i, part in enumerate(character.char_tree_array):
        layer_info = character.layers_info[part]
        parent = None if character.parents[i] is None else parts_list[character.parents[i]]
        parts_list.append(BodyPart(parent, layer_info['name'], layer_info['path'], layer_info['displacement'],
                                   parameters[i]))
    return parts_list[0]


def load_data(char_name, batch_size=4, samples_num=10000, angle_range=15):
    try:
        char = Character(PATH + 'Character Layers\\' + char_name + '\\Config.txt')
    except OSError:
        print("Couldn't find config file for " + char_name)
        return None
    num_layers = len(char.char_tree_array)
    labels = np.random.randint(-angle_range, angle_range, size=samples_num * num_layers).reshape((samples_num, num_layers))
    data = []
    im_batch = []
    label_batch = []
    i = 1
    for index in tqdm(range(len(labels))):
        angles = labels[index]
        im = create_image(char, angles, draw_skeleton=False,
                          print_dict=False, as_image=False)
        im = (im - 127.5) / 127.5
        im_batch.append(im)
        label_batch.append(angles)
        if i % batch_size == 0:
            data.append((torch.tensor(np.array(im_batch, dtype='float64')),
                         torch.tensor(np.array(label_batch, dtype='float64'))))
            im_batch = []
            label_batch = []
        i += 1

    print("finished forging data")
    cutoff = int(len(data) * 0.8)
    return data[:cutoff], data[cutoff:], char


if __name__ == "__main__":
    # char = Character()
    # json.dump(char, open(char.path + 'Config', 'w'), default=lambda o: o.__dict__,
    #         sort_keys=True, indent=4)
    load_data('Cartman', batch_size=4, samples_num=25)
    # Character.create_default_config_file(PATH + 'Character Layers\\Default Character\\', 'Lower Torso',
    #                                      {'Root': ['Chest', 'Upper Left Leg', 'Upper Right Leg'],
    #                                       'Chest': ['Head', 'Left Shoulder', 'Right Shoulder'],
    #                                       'Left Shoulder': ['Left Arm'],
    #                                       'Right Shoulder': ['Right Arm'],
    #                                       'Upper Left Leg': ['Lower Left Leg'],
    #                                       'Upper Right Leg': ['Lower Right Leg'],
    #                                       },
    #                                      ["Root",
    #                                       "Upper Left Leg",
    #                                       "Lower Left Leg",
    #                                       "Upper Right Leg",
    #                                       "Lower Right Leg",
    #                                       "Chest",
    #                                       "Head",
    #                                       "Left Shoulder",
    #                                       "Left Arm",
    #                                       "Right Shoulder",
    #                                       "Right Arm"])
    #
    # char = Character(PATH + 'Character Layers\\Aang\\Config.txt')
    # angles = char.sample_params
    # im = create_image(char, angles, draw_skeleton=True, print_dict=False, as_image=True)
    # im.show()
#
