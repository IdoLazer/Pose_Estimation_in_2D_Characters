import json
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

import DataModule
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

    def __mul__(self, other):
        if isinstance(other, Vector2D):
            x = self.x * other.x
            y = self.y * other.y
        else:
            x = self.x * other
            y = self.y * other
        return Vector2D(x, y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, other):
        if isinstance(other, Vector2D):
            self.x *= other.x
            self.y *= other.y
        else:
            self.x *= other
            self.y *= other
        return self

    def __floordiv__(self, other):
        if isinstance(other, Vector2D):
            self.x /= other.x
            self.y /= other.y
        else:
            self.x /= other
            self.y /= other
        return self

    def __str__(self):
        return "({0},{1})".format(self.x, self.y)


images = dict()
colored_images = dict()


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
    def __init__(self, parent, name, path: str, dist_from_parent, parameters):
        self.parent = parent
        self.name = name
        if name not in images and path is not None:
            self.im = Image.open(path)
            images[name] = self.im
            self.colored_im = Image.open(path.split('.')[0] + '_color.png')
            colored_images[name] = self.colored_im
        elif path is not None:
            self.im = images[name]
            self.colored_im = colored_images[name]
        inner_rotation, x_scaling, y_scaling, x_translate, y_translate = parameters
        translation = Vector2D(x_translate, y_translate)
        scaling = Vector2D(x_scaling, y_scaling)
        joint_rotation = math.radians(0)
        center = Vector2D() + Vector2D(x_translate, y_translate)
        if parent is not None:
            parent.__add_child(self)
            joint_rotation = parent.rotation
            center = parent.position + translation
            dist_from_parent = dist_from_parent * parent.scaling
            # scaling.x = scaling.x / parent.scaling.x
            # scaling.y = scaling.y / parent.scaling.y
            # scaling.x = np.min([np.max([scaling.x * parent.scaling.x, config['dataset']['scaling_range'][0]]),
            #                   config['dataset']['scaling_range'][1]])
            # scaling.y = np.min([np.max([scaling.y * parent.scaling.y, config['dataset']['scaling_range'][0]]),
            #                   config['dataset']['scaling_range'][1]])
        # create_affine_transform(math.radians(inner_rotation), Vector2D(), dist_from_parent, Vector2D(x_scaling, y_scaling),
        #                         name, True, self.im.size[0])  # TODO: This is just to generate initial transformations
        self.position = rotate(center, center + dist_from_parent, -joint_rotation)
        self.center = center
        self.rotation = joint_rotation + math.radians(inner_rotation)
        self.scaling = scaling
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
    sx, sy = scaling.x, scaling.y
    a = cos / sx
    b = sin / sx
    c = x - a * nx - b * ny
    d = -sin / sy
    e = cos / sy
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


def traverse_tree(cur, size, layers, matrices, draw_skeleton, skeleton_draw, transform=True, print_dict=False,
                  colored=False):
    image_center = Vector2D(size / 2, size / 2)
    im = cur.colored_im if colored else cur.im
    if transform:
        transform_matrix = create_affine_transform(cur.rotation, image_center, cur.position, cur.scaling, cur.name,
                                                   for_label=True, im_size=size)
        matrices[cur.name] = transform_matrix
        im = im.transform((size, size),
                                  Image.AFFINE,
                                  data=create_affine_transform(cur.rotation, image_center, cur.position,
                                                               cur.scaling, cur.name, print_dict).flatten()[:6],
                                  resample=Image.BILINEAR)
    for child in cur.children:
        if draw_skeleton:
            line = translate_points([cur.position, child.position], image_center)
            skeleton_draw.ellipse(
                xy=(line[1].x - 1, size - (line[1].y + 1), line[1].x + 1, size - (line[1].y - 1),),
                fill="red")
            skeleton_draw.line([(line[0].x, size - line[0].y),
                                (line[1].x, size - line[1].y)], fill="yellow")
        traverse_tree(child, size, layers, matrices, draw_skeleton, skeleton_draw, transform, print_dict, colored)
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


def generate_layers(character, parameters, draw_skeleton=False, as_tensor=False, transform=True, print_dict=False,
                    colored=False):
    origin = create_body_hierarchy(parameters, character)
    layers = {}
    matrices = {}
    skeleton = Image.new('RGBA', (character.image_size, character.image_size))
    skeleton_draw = ImageDraw.Draw(skeleton)
    traverse_tree(origin, character.image_size, layers, matrices, draw_skeleton, skeleton_draw, transform, print_dict,
                  colored)
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


def create_image(character, parameters, draw_skeleton=False, print_dict=False, as_image=False, random_order=True):
    drawing_order = character.drawing_order + ['Skeleton']
    layers, transformations = generate_layers(character, parameters, draw_skeleton, print_dict=print_dict)
    im_size = character.image_size
    im = Image.new("RGBA", (im_size, im_size))
    matrix_list = []
    for part in character.char_tree_array:
        transform_matrix = transformations.get(part, np.array([1, 0, 0, 0, 1, 0]))
        matrix_list.append(transform_matrix)

    # if random_order:
    #     rand_int = np.random.randint(config['dataset']['max_layer_swaps'])
    #     for rand in range(rand_int):
    #         i = np.random.randint(len(drawing_order) - 1)
    #         j = np.random.randint(len(drawing_order) - 1)
    #         drawing_order[j], drawing_order[i] = drawing_order[i], drawing_order[j]

    for part in drawing_order:
        alpha = ImageOps.invert(layers[part].split()[-1])
        im = Image.composite(im, layers[part], alpha)
    return (im, matrix_list) if as_image else (np.array(im).astype('uint8'), np.array(matrix_list))


def create_body_hierarchy(parameters, character):
    if parameters is not None:
        angles = parameters[0]
        for i in range(len(angles)):
            angles[i] += character.sample_params[i]
        # parameters[0] = [0, 0, 50, 60, 0, -10, 60, 60, -5, 10, -30, -30, 30, -30]  # TODO: Always comment out when starting
        parameters = parameters.transpose()
        new_params = np.array([0, 1, 1, 0, 0] * len(character.char_tree_array), dtype=float).\
            reshape((len(character.char_tree_array), 5))
        new_params[2] = parameters[2]
        new_params[3] = parameters[3]
        new_params[4] = parameters[4]
        new_params[5] = parameters[5]
        new_params[6] = parameters[6]
        new_params[7] = parameters[7]
        new_params[8] = parameters[8]
        new_params[9] = parameters[9]
        parameters = new_params
    else:
        parameters = np.array([0, 1, 1, 0, 0] * len(character.char_tree_array)).\
            reshape((len(character.char_tree_array), 5))
    parts_list = []
    for i, part in enumerate(character.char_tree_array):
        layer_info = character.layers_info[part]
        parent = None if character.parents[i] is None else parts_list[character.parents[i]]
        parts_list.append(BodyPart(parent, layer_info['name'], layer_info['path'], layer_info['displacement'],
                                   parameters[i]))
    return parts_list[0]


if __name__ == "__main__":
    # char = Character()
    # json.dump(char, open(char.path + 'Config', 'w'), default=lambda o: o.__dict__,
    #         sort_keys=True, indent=4)
    # Character.create_default_config_file(config['dirs']['source_dir'] + 'Character Layers\\Aang2\\', 'Torso',
    #                                      {'Root': ['Head', 'Left Shoulder', 'Right Shoulder', 'Left Upper Leg', 'Right Upper Leg'],
    #                                       'Left Shoulder': ['Left Arm'],
    #                                       'Left Arm': ['Left Palm'],
    #                                       'Right Shoulder': ['Right Arm'],
    #                                       'Right Arm': ['Right Palm'],
    #                                       'Left Upper Leg': ['Left Lower Leg'],
    #                                       'Left Lower Leg': ['Left Foot'],
    #                                       'Right Upper Leg': ['Right Lower Leg'],
    #                                       'Right Lower Leg': ['Right Foot'],
    #                                       },
    #                                      ["Head",
    #                                       "Root",
    #                                       "Left Foot",
    #                                       "Left Lower Leg",
    #                                       "Left Upper Leg",
    #                                       "Right Foot",
    #                                       "Right Lower Leg",
    #                                       "Right Upper Leg",
    #                                       "Left Palm",
    #                                       "Left Arm",
    #                                       "Left Shoulder",
    #                                       "Right Palm",
    #                                       "Right Arm",
    #                                       "Right Shoulder"])

    char = Character(config['dirs']['source_dir'] + 'Character Layers\\Aang2\\Config.txt')
    parameters = DataModule.generate_parameters(len(char.char_tree_array), 1)
    im, mat = create_image(char, parameters[0], draw_skeleton=False, print_dict=False, as_image=True, random_order=False)
    im.show()
    im.save(config['dirs']['source_dir'] + 'Test Inputs\\Images\\fabricated_post.png')
