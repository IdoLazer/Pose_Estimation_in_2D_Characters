import json
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from Config import config


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
            char_conf = json.load(open(path, 'r'))
            self.image_size = char_conf['image_size']
            self.path = char_conf['path']
            self.char_tree = char_conf['char_tree']
            self.layers_info = char_conf['layers_info']
            for key in self.layers_info:
                item = self.layers_info[key]
                item['displacement'] = Vector2D(item['displacement'][0], item['displacement'][1])
            self.char_tree_array = char_conf['char_tree_array']
            self.sample_params = char_conf['sample_params']
            self.drawing_order = char_conf['drawing_order']
            self.parents = char_conf['parents']
            self.canonical_bias_dict = char_conf['canonical_bias_dict']

    @staticmethod
    def create_default_config_file(path, root_name, char_tree, drawing_order):
        default_char = Character()

        # find image size
        default_char.image_size = Image.open(path + root_name + '.png').size[0]
        default_char.path = path
        default_char.char_tree = char_tree
        default_char.layers_info = dict()
        char_tree_array = ["Root"]
        index = 0
        parents = [None]
        while index < len(char_tree_array):
            part = char_tree_array[index]
            default_char.layers_info[part] = \
                {
                    "displacement": [0, 0],
                    "name": part,
                    "path": path + (root_name if part == "Root" else part) + '.png'
                }
            if part in char_tree:
                char_tree_array += char_tree[part]
                parents += [index] * len(char_tree[part])
            index += 1
        default_char.char_tree_array = char_tree_array
        default_char.drawing_order = drawing_order
        default_char.parents = parents
        default_char.sample_params = [0] * len(default_char.char_tree_array)
        json.dump(default_char, open(default_char.path + 'Config.txt', 'w'), default=lambda o: o.__dict__,
                  sort_keys=True, indent=4)


class BodyPart:
    def __init__(self, parent, name, path, dist_from_parent, parameters):
        self.parent = parent
        self.name = name
        if name not in images and path is not None:
            self.im = Image.open(path)
            images[name] = self.im
        elif path is not None:
            self.im = images[name]
        inner_rotation, x_scaling, y_scaling, x_translate, y_translate = parameters
        joint_rotation = math.radians(0)
        center = Vector2D() + Vector2D(x_translate, y_translate)
        if parent is not None:
            parent.__add_child(self)
            joint_rotation = parent.rotation
            center = parent.position + Vector2D(x_translate, y_translate)
        # create_affine_transform(math.radians(inner_rotation), Vector2D(), dist_from_parent,
        #                         name, True, self.im.size[0])  # TODO: This is just to generate initial transformations
        self.position = rotate(center, center + dist_from_parent, -joint_rotation)
        self.center = center
        self.rotation = joint_rotation + math.radians(inner_rotation)
        self.scaling = Vector2D(x_scaling, y_scaling)
        self.children = []

    def __add_child(self, body_part):
        self.children.append(body_part)


try:
    char = Character(config['dirs']['source_dir'] + 'Character Layers\\' + config['dataset']['character'] +
                     '\\Config.txt')
except OSError:
    print("Couldn't find config file for " + config['dataset']['character'])


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


def create_affine_transform(angle, center, displacement, scaling, name, print_dict=False, for_label=False,
                            im_size=None):
    dx, dy = (displacement.x, -displacement.y) if print_dict else \
        (displacement.x, -displacement.y)
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y = (0, 0) if (print_dict or for_label) else (center.x, center.y)
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
    if for_label and im_size is not None:
        return [a, b, c / (im_size / 2), d, e, f / (im_size / 2)]
    return np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1]])


def traverse_tree(cur, size, layers, matrices, draw_skeleton, skeleton_draw, transform=True, print_dict=False):
    image_center = Vector2D(size / 2, size / 2)
    im = cur.im
    if transform:
        # Scale layer according to scaling parameter
        im = im.resize((int(size * cur.scaling.x), int(size * cur.scaling.y)), resample=Image.BILINEAR)
        if cur.scaling.x < 1:
            left = 0
            right = int(size * cur.scaling.x)
        else:
            left = (int(size * cur.scaling.x) / 2) - (size / 2)
            right = (int(size * cur.scaling.x) / 2) + (size / 2)
        if cur.scaling.y < 1:
            top = 0
            down = int(size * cur.scaling.y)
        else:
            top = (int(size * cur.scaling.y) / 2) - (size / 2)
            down = (int(size * cur.scaling.y) / 2) + (size / 2)
        im = im.crop((left, top, right, down))
        img_w, img_h = im.size
        background = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(im, offset)
        # transform scaled layer according to other parameters
        transform_matrix = create_affine_transform(cur.rotation, image_center, cur.position, cur.scaling, cur.name,
                                                   for_label=True, im_size=size)
        matrices[cur.name] = transform_matrix
        im = background.transform((size, size),
                          Image.AFFINE,
                          data=create_affine_transform(cur.rotation, image_center, cur.position, cur.scaling, cur.name,
                                                       print_dict).flatten()[:6],
                          resample=Image.BILINEAR)
    for child in cur.children:
        if draw_skeleton:
            line = translate_points([cur.position, child.position], image_center)
            skeleton_draw.ellipse(
                xy=(line[1].x - 1, size - (line[1].y + 1), line[1].x + 1, size - (line[1].y - 1),),
                fill="red")
            skeleton_draw.line([(line[0].x, size - line[0].y),
                                (line[1].x, size - line[1].y)], fill="yellow")
        traverse_tree(child, size, layers, matrices, draw_skeleton, skeleton_draw, transform, print_dict)
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


def generate_layers(character, parameters, draw_skeleton=False, as_tensor=False, transform=True, print_dict=False):
    origin = create_body_hierarchy(parameters, character)
    layers = {}
    matrices = {}
    skeleton = Image.new('RGBA', (character.image_size, character.image_size))
    skeleton_draw = ImageDraw.Draw(skeleton)
    traverse_tree(origin, character.image_size, layers, matrices, draw_skeleton, skeleton_draw, transform, print_dict)
    if not as_tensor:
        layers['Skeleton'] = skeleton
        return layers, matrices
    layers_list = []
    matrix_list = []
    for part in character.char_tree_array:

        transform_matrix = matrices.get(part, [[1, 0, 0], [0, 1, 0]])
        matrix_list.append(transform_matrix)

        im = Image.new("RGBA", (character.image_size, character.image_size))
        alpha = ImageOps.invert(layers[part].split()[-1])
        layer = Image.composite(im, layers[part], alpha)
        # layer = layer.convert("RGB")
        layer = np.array(layer)
        layer = (layer - 127.5) / 127.5
        layers_list.append(layer)
    return torch.tensor(np.array(layers_list, dtype='float64')), torch.tensor(np.array(matrix_list, dtype='float64'))


def create_image(character, parameters, draw_skeleton=False, omit_layers=False, print_dict=False, as_image=False):
    drawing_order = character.drawing_order + ['Skeleton']
    layers, transformations = generate_layers(character, parameters, draw_skeleton, print_dict=print_dict)
    im_size = character.image_size
    im = Image.new("RGBA", (im_size, im_size))
    matrix_list = []
    for part in character.char_tree_array:
        transform_matrix = transformations.get(part, np.array([1, 0, 0, 0, 1, 0]))
        matrix_list.append(transform_matrix)

    for part in drawing_order:
        if omit_layers and part != 'Right Arm':
            continue
        alpha = ImageOps.invert(layers[part].split()[-1])
        im = Image.composite(im, layers[part], alpha)
    return (im.convert("RGB"), matrix_list) if as_image else (np.array(im).astype('uint8'), np.array(matrix_list))


def create_body_hierarchy(parameters, character):
    if parameters is not None:
        angles = parameters[0]
        for i in range(len(angles)):
            angles[i] += character.sample_params[i]
        # parameters[0] = [5, 14, -29, -16, 22, -10, -22, 14, -21, -26]  # TODO: Always comment out when starting
        parameters = parameters.transpose()
    else:
        parameters = np.array([0, 0, 0, 1, 1] * len(character.char_tree_array)).\
            reshape((len(character.char_tree_array), 5))
    parts_list = []
    for i, part in enumerate(character.char_tree_array):
        layer_info = character.layers_info[part]
        parent = None if character.parents[i] is None else parts_list[character.parents[i]]
        parts_list.append(BodyPart(parent, layer_info['name'], layer_info['path'], layer_info['displacement'],
                                   parameters[i]))
    return parts_list[0]


def generate_parameters(angle_range, num_layers, samples_num):
    angles = np.random.randint(-angle_range, angle_range, size=samples_num * num_layers). \
        reshape((samples_num, 1, num_layers))
    x_scaling = np.random.uniform(0.8, 1.2, size=samples_num * num_layers). \
        reshape((samples_num, 1, num_layers))
    y_scaling = np.random.uniform(0.8, 1.2, size=samples_num * num_layers). \
        reshape((samples_num, 1, num_layers))
    x_translate = np.random.randint(-1, 1, size=samples_num * num_layers). \
        reshape((samples_num, 1, num_layers))
    y_translate = np.random.randint(-1, 1, size=samples_num * num_layers). \
        reshape((samples_num, 1, num_layers))
    parameters = np.concatenate((angles, x_scaling / x_scaling, y_scaling / y_scaling, x_translate * 0, y_translate * 0), axis=1)
    return parameters


def load_data(batch_size=4, samples_num=100, angle_range=15):
    num_layers = len(char.char_tree_array)
    labels = generate_parameters(angle_range, num_layers, samples_num)

    data = []
    im_batch = []
    label_batch = []
    i = 1
    for index in tqdm(range(len(labels))):
        parameters = labels[index]
        im, matrices = create_image(char, parameters, draw_skeleton=False,
                          print_dict=False, as_image=False)
        im = (im - 127.5) / 127.5
        im_batch.append(im)
        label_batch.append(matrices)
        if i % batch_size == 0:
            data.append((torch.tensor(np.array(im_batch, dtype='float64')),
                         torch.tensor(np.array(label_batch, dtype='float64'))))
            im_batch = []
            label_batch = []
        i += 1

    print("finished forging data")
    return data


if __name__ == "__main__":
    # char = Character()
    # json.dump(char, open(char.path + 'Config', 'w'), default=lambda o: o.__dict__,
    #         sort_keys=True, indent=4)
    load_data(batch_size=4, samples_num=25, angle_range=15)
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
    # char = Character(config['dirs']['source_dir'] + 'Character Layers\\Aang2\\Config.txt')
    # parameters = generate_parameters(45, len(char.char_tree_array), 1)
    # im = create_image(char, parameters[0], draw_skeleton=False, print_dict=False, as_image=True)
    # im.show()
