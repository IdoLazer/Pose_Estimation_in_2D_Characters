import json_tricks as json
import ImageGenerator
from ImageGenerator import Vector2D
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def get_parameters_from_annotations(char, im_annotations):
    part_rotations = [0] * len(char.char_tree_array)
    part_inherited_rotations = [0] * len(char.char_tree_array)
    part_translations = [[0, 0]] * len(char.char_tree_array)
    for i, part in enumerate(char.char_tree_array):
        layer_info = char.layers_info[part]
        parent = char.parents[i]
        joint = im_annotations[i]
        joint = Vector2D(joint[0], -joint[1]) - Vector2D(char.image_size // 2, -char.image_size // 2)
        if parent is None:
            translation = joint - layer_info['displacement']
            part_translations[i] = [translation.x, translation.y]
        else:
            parent_joint = im_annotations[parent]
            parent_joint = Vector2D(parent_joint[0], -parent_joint[1]) - Vector2D(char.image_size // 2, -char.image_size // 2)
            v1 = layer_info['displacement']
            parent_rotation = part_inherited_rotations[parent]
            v1 = ImageGenerator.rotate(Vector2D(0, 0), v1, -math.radians(parent_rotation))
            v2 = (joint - parent_joint)
            if parent != 0:
                n1 = v1.normalized()
                n2 = v2.normalized()
                cos = n1.dot(n2)
                sin = n1.cross(n2)
                angle = -math.degrees(math.atan2(sin, cos))
                joint_supposed_pos = parent_joint + v2.normalized() * v1.size()
            else:
                angle = 0
                joint_supposed_pos = parent_joint + v1

            translation = joint - joint_supposed_pos

            part_rotations[parent] = angle
            part_inherited_rotations[i] = angle + part_inherited_rotations[parent]
            part_translations[i] = [translation.x, translation.y]
    part_translations = np.array(part_translations)
    part_translations = np.transpose(part_translations)
    parameters = [part_rotations,
                  np.ones(len(char.char_tree_array)),
                  np.ones(len(char.char_tree_array)),
                  part_translations[0],
                  part_translations[1]]
    return np.array(parameters)


def main(filename):
    scale = 1.5
    char = ImageGenerator.char
    folder = r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\Project Files\Sequences\Front"
    im_file_names = []
    with open(filename) as f:
        all_annotations = json.load(f)
        for i in range(len(all_annotations)):
            im_annotations = np.array([joint for joint in all_annotations[str(i)].values()]) * scale - (scale - 1) * 64
            parameters = get_parameters_from_annotations(char, im_annotations)
            im, data = ImageGenerator.create_image(char, parameters, as_image=False, random_order=False, random_generation=False)
            joints = np.array(data['joints'])
            joints = np.transpose(joints)
            fig = plt.figure()
            plt.imshow(im)
            im_annotations = np.transpose(im_annotations)
            plt.scatter(im_annotations[0], im_annotations[1])
            plt.scatter(joints[0], joints[1])
            im_file_name = f"{folder}\\im{i}.png"
            im_file_names.append(im_file_name)
            plt.savefig(im_file_name)

            plt.close(fig)

        # build gif
        with imageio.get_writer(f'{folder}\\mygif.gif', mode='I') as writer:
            for filename in im_file_names:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(im_file_names):
            os.remove(filename)


if __name__ == "__main__":
    main(r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\pose_estimation\output\aang\pose_resnet_16\128x128_d256x3_adam_lr1e-3\val\val_file\_iter_0_joints_pred.json")