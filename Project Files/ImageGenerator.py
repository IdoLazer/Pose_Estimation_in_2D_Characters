import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

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


IMAGE_SIZE = 128
CHARACTER_PATH = PATH + 'Character Layers\\'
CHARACTER_DICT = {
    # 'Lower Torso': {'name': 'Lower Torso', 'path': CHARACTER_PATH + 'Character.png', 'displacement': Vector2D(0, 8),
    #                 'children': ['Chest', 'Left Shoulder', 'Upper Right Leg'], 'parent': 'None'},
    'Lower Torso': {'name': 'Lower Torso', 'path': CHARACTER_PATH + 'Lower Torso.png', 'displacement': Vector2D(0, 8),
                    'children': ['Chest', 'Left Shoulder', 'Upper Right Leg'], 'parent': 'None'},
    'Chest': {'name': 'Chest', 'path': CHARACTER_PATH + 'Chest.png', 'displacement': Vector2D(0, 7),
              'children': ['Head', 'Left Shoulder', 'Right Shoulder'], 'parent': 'Lower Torso'},
    'Head': {'name': 'Head', 'path': CHARACTER_PATH + 'Head.png', 'displacement': Vector2D(0, 20),
             'children': [], 'parent': 'Chest'},
    'Left Shoulder': {'name': 'Left Shoulder', 'path': CHARACTER_PATH + 'Left Shoulder.png',
                      'displacement': Vector2D(-10, 13),
                      'children': ['Left Arm'], 'parent': 'Chest'},
    'Left Arm': {'name': 'Left Arm', 'path': CHARACTER_PATH + 'Left Arm.png', 'displacement': Vector2D(-6, -15),
                 'children': [], 'parent': 'Left Shoulder'},
    'Right Shoulder': {'name': 'Right Shoulder', 'path': CHARACTER_PATH + 'Right Shoulder.png',
                       'displacement': Vector2D(11, 13),
                       'children': ['Right Arm'], 'parent': 'Chest'},
    'Right Arm': {'name': 'Right Arm', 'path': CHARACTER_PATH + 'Right Arm.png', 'displacement': Vector2D(4, -16),
                  'children': [], 'parent': 'Right Shoulder'},
    'Upper Left Leg': {'name': 'Upper Left Leg', 'path': CHARACTER_PATH + 'Upper Left Leg.png',
                       'displacement': Vector2D(-5, -8),
                       'children': ['Lower Left Leg'], 'parent': 'Lower Torso'},
    'Lower Left Leg': {'name': 'Lower Left Leg', 'path': CHARACTER_PATH + 'Lower Left Leg.png',
                       'displacement': Vector2D(0, -28),
                       'children': [''], 'parent': 'Upper Left Leg'},
    'Upper Right Leg': {'name': 'Upper Right Leg', 'path': CHARACTER_PATH + 'Upper Right Leg.png',
                        'displacement': Vector2D(5, -8),
                        'children': ['Lower Right Leg'], 'parent': 'Lower Torso'},
    'Lower Right Leg': {'name': 'Lower Right Leg', 'path': CHARACTER_PATH + 'Lower Right Leg.png',
                        'displacement': Vector2D(1, -26),
                        'children': [''], 'parent': 'Upper Right Leg'},

}
DRAWING_ORDER = ['Lower Torso', 'Upper Left Leg', 'Lower Left Leg', 'Upper Right Leg', 'Lower Right Leg', 'Chest',
                 'Head', 'Left Shoulder', 'Left Arm', 'Right Shoulder', 'Right Arm']
PARENTS = [None, 0, 1, 0, 3, 0, 5, 5, 7, 5, 9]

images = dict()

# sample_params = [5, 14, -29, -16, 22, -10, -22, 14, -21, -26, -21]
sample_params = [5, 14, -29, -16, 22, -10, -22, 14, -21, -26, -21]


class BodyPart:
    def __init__(self, parent, name, path, shape, dist_from_parent, inner_rotation, color):
        self.parent = parent
        self.name = name
        self.im = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE))
        if name not in images and path is not None:
            self.im = Image.open(path)
            images[name] = self.im
        elif path is not None:
            self.im = images[name]
        elif shape is not None and color is not None:
            draw = ImageDraw.Draw(self.im)
            image_center = Vector2D(IMAGE_SIZE / 2, IMAGE_SIZE / 2)
            shape = translate_points(shape, image_center)
            draw.polygon([(shape[i].x, IMAGE_SIZE - shape[i].y) for i in range(len(shape))], fill=color,
                         outline=color)
        joint_rotation = math.radians(0)
        center = Vector2D()
        if parent is not None:
            parent.__add_child(self)
            joint_rotation = parent.rotation
            center = parent.position
        # create_affine_transform(math.radians(inner_rotation), Vector2D(), dist_from_parent,
        #                         name, True)  # TODO: This is just to generate initial transformations
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


def create_affine_transform(angle, center, displacement, name, print_dict=False):
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
        print('\'' + name + '\' : ' + str(bias) + ',')
    return np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1]])


def traverse_tree(cur, layers, draw_skeleton, skeleton_draw, transform=True, print_dict=False):
    image_center = Vector2D(IMAGE_SIZE / 2, IMAGE_SIZE / 2)
    im = cur.im
    if transform:
        im = im.transform((IMAGE_SIZE, IMAGE_SIZE),
                          Image.AFFINE,
                          data=create_affine_transform(cur.rotation, image_center, cur.position, cur.name, print_dict).
                          flatten()[:6],
                          resample=Image.BILINEAR)
    for child in cur.children:
        if draw_skeleton:
            line = translate_points([cur.position, child.position], image_center)
            skeleton_draw.ellipse(
                xy=(line[1].x - 1, IMAGE_SIZE - (line[1].y + 1), line[1].x + 1, IMAGE_SIZE - (line[1].y - 1),),
                fill="red")
            skeleton_draw.line([(line[0].x, IMAGE_SIZE - line[0].y),
                                (line[1].x, IMAGE_SIZE - line[1].y)], fill="yellow")
        traverse_tree(child, layers, draw_skeleton, skeleton_draw, transform, print_dict)
    if not cur.children and draw_skeleton:
        if cur.name == 'Head':
            line = translate_points([cur.position, rotate(cur.position, cur.position + Vector2D(0, 15), -cur.rotation)],
                                    image_center)
        else:
            line = translate_points(
                [cur.position, rotate(cur.position, cur.position + Vector2D(0, -15), -cur.rotation)], image_center)
        skeleton_draw.line([(line[0].x, IMAGE_SIZE - line[0].y),
                            (line[1].x, IMAGE_SIZE - line[1].y)], fill="yellow")
    layers[cur.name] = im


def generate_layers(origin, draw_skeleton=False, as_tensor=False, transform=True, print_dict=False):
    layers = {}
    skeleton = Image.new('RGBA', (IMAGE_SIZE, IMAGE_SIZE))
    skeleton_draw = ImageDraw.Draw(skeleton)
    traverse_tree(origin, layers, draw_skeleton, skeleton_draw, transform, print_dict)
    if not as_tensor:
        layers['Skeleton'] = skeleton
        return layers
    layers_list = []
    for part in DRAWING_ORDER:
        im = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE))
        alpha = ImageOps.invert(layers[part].split()[-1])
        layer = Image.composite(im, layers[part], alpha)
        # layer = layer.convert("RGB")
        layer = np.array(layer)
        layer = (layer - 127.5) / 127.5
        layers_list.append(layer)
    return torch.tensor(np.array(layers_list, dtype='float64'))


# def transform_image(layers, transforms):
#     new_layers = []
#     for i in range(len(layers)):
#         new_layers.append(np.asarray(layers[i].transform(
#             (IMAGE_SIZE, IMAGE_SIZE),
#             Image.AFFINE,
#             data=transforms[i].flatten()[:6],
#             resample=Image.BILINEAR)))
#     im = np.array(new_layers).sum(axis=0).astype('uint8')
#     return im


def create_image(origin, draw_skeleton=False, omit_layers=False, print_dict=False):
    drawing_order = DRAWING_ORDER + ['Skeleton']
    layers = generate_layers(origin, draw_skeleton, print_dict=print_dict)
    # new_layers = []
    im = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE))
    for part in drawing_order:
        if omit_layers and part != 'Right Arm':
            continue
        alpha = ImageOps.invert(layers[part].split()[-1])
        im = Image.composite(im, layers[part], alpha)
    # return np.array(new_layers).sum(axis=0).astype('uint8')
    # im = im.convert("RGB")
    # im.show()
    return np.array(im).astype('uint8')


def create_body_hierarchy(parameters):
    # print("sample_params = " + str(parameters))
    for i in range(len(parameters)):
        parameters[i] += sample_params[i]
    # parameters = sample_params  # TODO: Always comment out when starting
    cur = CHARACTER_DICT['Lower Torso']
    parts_dict = dict()
    parts_dict['None'] = None
    for i, part in enumerate(DRAWING_ORDER):
        cur = CHARACTER_DICT[part]
        parts_dict[part] = BodyPart(parts_dict[cur['parent']], cur['name'], cur['path'], None,
                                    cur['displacement'], parameters[i], None)
    return parts_dict['Lower Torso']


def load_data(batch_size=4):
    samples_num = 10000
    num_layers = len(CHARACTER_DICT)
    labels = np.random.randint(-35, 35, size=samples_num * num_layers).reshape((samples_num, num_layers))
    data = []
    im_batch = []
    label_batch = []
    i = 1
    for index in tqdm(range(len(labels))):
        angles = labels[index]
        origin = create_body_hierarchy(angles)
        im = create_image(origin, draw_skeleton=False, print_dict=False)#, omit_layers=(index // batch_size) % 2 == 1)
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
    return data[:cutoff], data[cutoff:]


if __name__ == "__main__":
    angles = np.random.randint(-10, 10, 11)
    origin = create_body_hierarchy(angles)
    im = create_image(origin, draw_skeleton=False)
    plt.imshow(im)
    plt.show()
