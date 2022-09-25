import json_tricks as json
import project_files.ImageGenerator
from project_files.ImageGenerator import Vector2D
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image


def get_parameters_from_annotations(char, im_annotations):
    part_rotations = [0] * len(char.char_tree_array)
    part_actual_rotations = [0] * len(char.char_tree_array)
    part_translations = [[0, 0]] * len(char.char_tree_array)
    parent_rotations = []
    curr_parent = None
    for i, part in enumerate(char.char_tree_array):
        layer_info = char.layers_info[part]
        parent = char.parents[i]
        joint = im_annotations[i]
        joint = Vector2D(joint[0], -joint[1]) - Vector2D(char.image_size // 2, -char.image_size // 2)

        if parent is None:
            translation = joint - layer_info['displacement']
            part_translations[i] = [translation.x, translation.y]

        else:
            if curr_parent is None:
                curr_parent = parent
            if curr_parent != parent:
                part_actual_rotations[curr_parent] = np.average(parent_rotations)
                parent_rotations = []
                curr_parent = parent

            parent_joint = im_annotations[parent]
            parent_joint = Vector2D(parent_joint[0], -parent_joint[1]) - Vector2D(char.image_size // 2, -char.image_size // 2)
            v1 = layer_info['displacement'].copy()
            # parent_rotation = part_actual_rotations[parent]
            # v1 = project_files.ImageGenerator.rotate(Vector2D(0, 0), v1, -math.radians(parent_rotation))
            v2 = (joint - parent_joint)
            # if parent != 0:
            #     n1 = v1.normalized()
            #     n2 = v2.normalized()
            #     cos = n1.dot(n2)
            #     sin = n1.cross(n2)
            #     angle = -math.degrees(math.atan2(sin, cos))
            #     joint_supposed_pos = parent_joint + v2.normalized() * v1.size()
            # else:
            #     angle = 0
            #     joint_supposed_pos = parent_joint + v1

            n1 = v1.normalized()
            n2 = v2.normalized()
            cos = n1.dot(n2)
            sin = n1.cross(n2)
            angle = -math.degrees(math.atan2(sin, cos))
            joint_supposed_pos = parent_joint + v2.normalized() * v1.size()

            translation = joint - joint_supposed_pos

            parent_rotations.append(angle)
            # part_actual_rotations[i] = angle  # + part_actual_rotations[parent]
            part_translations[i] = [translation.x, translation.y]

    for i, parent in enumerate(char.parents):
        part_rotations[i] = part_actual_rotations[i] if parent is None else part_actual_rotations[i] - part_rotations[parent]
    part_translations = np.array(part_translations)
    part_translations = np.transpose(part_translations)
    parameters = [part_rotations,
                  np.ones(len(char.char_tree_array)),
                  np.ones(len(char.char_tree_array)),
                  part_translations[0],
                  part_translations[1]]
    return np.array(parameters)


def GenerateSequence(filename, scale=1):
    char_front = project_files.ImageGenerator.char
    char_side = project_files.ImageGenerator.char_side
    folder = '\\'.join(filename.split('\\')[:-1])
    name = filename.split('\\')[-1].split('.')[0]
    test_inputs_folder = r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\project_files\Test Inputs\Aang"
    im_file_names_annotated = []
    im_file_names = []
    with open(filename) as f:
        all_annotations = json.load(f)
        for i in range(len(all_annotations)):
            im_annotations = np.array([joint for joint in all_annotations[str(i)].values()]) * scale - (scale - 1) * 64
            front_parameters = get_parameters_from_annotations(char_front, im_annotations)
            side_parameters = get_parameters_from_annotations(char_side, im_annotations)
            if np.linalg.norm(front_parameters) < np.linalg.norm(side_parameters):
                char = char_front
                parameters = front_parameters
            else:
                char = char_side
                parameters = side_parameters
            im, data = project_files.ImageGenerator.create_image(char, parameters, as_image=False, random_order=False, random_generation=False)
            joints = np.array(data['joints'])
            joints = np.transpose(joints)

            test_im = Image.open(f"{test_inputs_folder}\\test_Pose{i+1}.png")
            test_im = np.asarray(test_im)

            im = np.concatenate([test_im, im[:, :, :3]], axis=1)

            fig = plt.figure()
            plt.imshow(im)

            im_file_name = f"{folder}\\im{i}.png"
            im_file_names.append(im_file_name)
            plt.savefig(im_file_name)

            im_annotations = np.transpose(im_annotations)
            plt.scatter(im_annotations[0], im_annotations[1])
            plt.scatter(im_annotations[0] + 128, im_annotations[1])

            im_file_name_annotated = f"{folder}\\im{i}_annot.png"
            im_file_names_annotated.append(im_file_name_annotated)
            plt.savefig(im_file_name_annotated)

            plt.close(fig)

        # build gif
        with imageio.get_writer(f'{folder}\\{name}_seq.gif', mode='I') as writer:
            for filename in im_file_names:
                image = imageio.imread(filename)
                writer.append_data(image)
            for filename in im_file_names_annotated:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(im_file_names):
            os.remove(filename)
        for filename in set(im_file_names_annotated):
            os.remove(filename)


if __name__ == "__main__":
    GenerateSequence(r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\pose_estimation\output\aang\pose_resnet_16\128x128_d256x3_adam_lr1e-3\val\08-02-16 24-08-2022 (new dataset, 4 layers)\epoch_24_iter_0_joints_pred.json")
