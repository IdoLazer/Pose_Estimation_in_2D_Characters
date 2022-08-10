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

    def __truediv__(self, other):
        if isinstance(other, Vector2D):
            x = self.x / other.x
            y = self.y / other.y
        else:
            x = self.x / other
            y = self.y / other
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

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def size(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalized(self):
        return self / self.size()


images_front = dict()
images_side = dict()
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
    def __init__(self, parent, name, path: str, dist_from_parent, parameters, images, flipped=0, is_random=True):
        self.parent = parent
        self.name = name
        self.mirror = np.random.randint(2) if is_random else 1
        self.flipped = False

        # Flipping right and left limbs
        if flipped == 1:
            self.flipped = True
            name = name.replace('Left', 'Right') if 'Left' in name else name.replace('Right', 'Left')

        # if name == "Root":
        #     self.mirror = np.random.randint(2)
        # else:
        #     self.mirror = 1

        # Randomly replacing front and side versions of 'edge' limbs
        if is_random and ('Foot' in name or 'Head' in name or 'Palm' in name):
            images_idx = np.random.randint(2)
            images = images_front if images_idx == 0 else images_side
            path = path.replace('Side', 'Front') if images_idx == 0 else path.replace('Front', 'Side')

        # Randomly mirroring limbs
        if name not in images and path is not None:
            im = Image.open(path)
            mirrored_im = Image.open(path.split('.')[0] + '_mirrored.png')
            images[name] = [mirrored_im, im]
            self.im = images[name][self.mirror]

            # colored_im = Image.open(path.split('.')[0] + '_color.png')
            # flipped_colored_im = ImageOps.mirror(colored_im)
            # colored_images[name] = [flipped_colored_im, colored_im]
            # self.colored_im = colored_images[name][self.mirror]

        elif path is not None:
            self.im = images[name][self.mirror]
            # self.colored_im = colored_images[name][self.mirror]

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
                     '\\Front\\Config.txt')
    char_side = Character(config['dirs']['source_dir'] + 'Character Layers\\' + config['dataset']['character'] +
                     '\\Side\\Config.txt')
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


def traverse_tree(cur, size, layers, joints, draw_skeleton, skeleton_draw, transform=True, print_dict=False,
                  colored=False):
    image_center = Vector2D(size / 2, size / 2)
    # im = cur.colored_im if colored else cur.im
    im = cur.im
    if transform:
        # transform_matrix = create_affine_transform(cur.rotation, image_center, cur.position, cur.scaling, cur.name,
        #                                            for_label=True, im_size=size)
        joints[cur.name] = [(cur.position + image_center).x, size - (cur.position + image_center).y]
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
        traverse_tree(child, size, layers, joints, draw_skeleton, skeleton_draw, transform, print_dict, colored)
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
                    colored=False, is_random=True):
    origin = create_body_hierarchy(parameters, character, is_random=is_random)
    layers = {}
    joints = {}
    skeleton = Image.new('RGBA', (character.image_size, character.image_size))
    skeleton_draw = ImageDraw.Draw(skeleton)
    traverse_tree(origin, character.image_size, layers, joints, draw_skeleton, skeleton_draw, transform, print_dict,
                  colored)
    if not as_tensor:
        layers['Skeleton'] = skeleton
        return layers, joints
    layers_list = []
    joints_list = []
    for part in character.char_tree_array:

        joint_pos = joints.get(part, [[1, 0, 0], [0, 1, 0]])
        joints_list.append(joint_pos)

        im = Image.new("RGBA", (character.image_size, character.image_size))
        alpha = ImageOps.invert(layers[part].split()[-1])
        layer = Image.composite(im, layers[part], alpha)
        # layer = layer.convert("RGB")
        layer = np.array(layer)
        layer = (layer - 127.5) / 127.5
        layers_list.append(layer)
    return torch.tensor(np.array(layers_list, dtype='float64')), torch.tensor(np.array(joints_list, dtype='float64'))


def create_image(character, parameters, draw_skeleton=False, print_dict=False, as_image=False, random_order=True, random_generation=True):
    drawing_order = character.drawing_order + ['Skeleton']
    layers, joints = generate_layers(character, parameters, draw_skeleton, print_dict=print_dict, is_random=random_generation)
    im_size = character.image_size
    im = Image.new("RGBA", (im_size, im_size))
    joints_list = []
    joints_vis_list = []
    for part in character.char_tree_array:
        joint_pos = joints.get(part, np.array([im_size // 2, im_size // 2]))
        joints_vis_list.append(1)
        joints_list.append(joint_pos)

    labels = {"joints_vis": joints_vis_list, "joints": joints_list, "scale": 1, "center": joints_list[0]}
    if random_order:
        rand_int = np.random.randint(config['dataset']['max_layer_swaps'])
        for rand in range(rand_int):
            i = np.random.randint(len(drawing_order) - 1)
            j = np.random.randint(len(drawing_order) - 1)
            drawing_order[j], drawing_order[i] = drawing_order[i], drawing_order[j]

    for part in drawing_order:
        alpha = ImageOps.invert(layers[part].split()[-1])
        im = Image.composite(im, layers[part], alpha)
    return (im, labels) if as_image else (np.array(im).astype('uint8'), labels)


def create_body_hierarchy(parameters, character, is_random=True):
    if parameters is not None:
        angles = parameters[0]
        for i in range(len(angles)):
            angles[i] += character.sample_params[i]
        # parameters[0] = [0, 0, 50, 60, 0, -10, 60, 60, -5, 10, -30, -30, 30, -30]  # TODO: Always comment out when starting
        parameters = parameters.transpose()
        if is_random:
            parameters[0][0] = np.random.randint(-20, 20)
        # new_params = np.array([0, 1, 1, 0, 0] * len(character.char_tree_array), dtype=float).\
        #     reshape((len(character.char_tree_array), 5))
        # l4 = [0, 10, 12]
        # l5 = [0, -10, -12]
        # l8 = [0, -8]
        # l9 = [0, 12]
        # l10 = [0, -80]
        # l11 = [0, -80]
        # new_params[2] = parameters[2]
        # new_params[3] = parameters[3]
        # new_params[4][0] = l4[np.random.randint(3)]
        # new_params[4][2] = 1
        # new_params[5][0] = l5[np.random.randint(3)]
        # new_params[5][2] = 1
        # new_params[6] = parameters[6]
        # new_params[7] = parameters[7]
        # new_params[8][0] = l8[np.random.randint(2)]
        # new_params[8][1] = 1.1
        # new_params[8][2] = 1.3
        # new_params[9][0] = l9[np.random.randint(2)]
        # new_params[9][2] = 1.4
        # new_params[10][0] = l10[np.random.randint(2)]
        # new_params[11][0] = l11[np.random.randint(2)]
        # new_params[12][0] = 15
        # new_params[12][1] = 1.5
        # new_params[12][2] = 0.8
        # new_params[13][0] = -10
        # new_params[5] = parameters[5]
        # parameters = new_params
    else:
        parameters = np.array([0, 1, 1, 0, 0] * len(character.char_tree_array)).\
            reshape((len(character.char_tree_array), 5))
    parts_list = []
    for i, part in enumerate(character.char_tree_array):
        layer_info = character.layers_info[part]
        name = layer_info['name']

        parent = None if character.parents[i] is None else parts_list[character.parents[i]]
        if parent is not None and parent.flipped:
            displacement_part = part.replace('Left', 'Right') if 'Left' in part else part.replace('Right', 'Left')
            displacement = character.layers_info[displacement_part]['displacement']
        else:
            displacement = layer_info['displacement']

        path = layer_info['path']
        if is_random and ('Left' in part or 'Right' in part):
            flipped = np.random.randint(2)
            # flipped = 0
            if flipped == 1:
                path_part = part.replace('Left', 'Right') if 'Left' in part else part.replace('Right', 'Left')
                path = character.layers_info[path_part]['path']
        else:
            flipped = 0
        # if flipped == 1:
        #     displacement *= Vector2D(-1, 1)
        images = images_front if character == char else images_side
        parts_list.append(BodyPart(parent, name, path, displacement, parameters[i], images, flipped, is_random=is_random))
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

    char = Character(config['dirs']['source_dir'] + 'Character Layers\\Aang2\\Front\\Config.txt')
    parameters = DataModule.generate_parameters(len(char.char_tree_array), 1, 50)
    im, mat = create_image(char, parameters[0], draw_skeleton=False, print_dict=False, as_image=False, random_order=False, random_generation=False)
    import matplotlib.pyplot as plt
    joints1 = np.array(mat['joints'])
    joints1 = np.transpose(joints1)
    plt.scatter(joints1[0], joints1[1])
    plt.imshow(im)
    plt.show()

    with open(r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\pose_estimation\output\aang\pose_resnet_16\128x128_d256x3_adam_lr1e-3\val\val_file\_iter_0_joints_pred.json") as f:
        all_annotations = json.load(f)
        x = 1.5
        joints2 = np.array([joint for joint in all_annotations['0'].values()]) * x - (x - 1) * 64
        joints2 = np.transpose(joints2)
        plt.scatter(joints2[0] + 3, joints2[1] - 9)
        plt.imshow(im)
        plt.show()
    # im.save(config['dirs']['source_dir'] + 'Test Inputs\\Images\\fabricated_post.png')
