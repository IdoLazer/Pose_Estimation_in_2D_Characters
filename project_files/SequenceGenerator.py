import json_tricks as json
import project_files.ImageGenerator
from project_files.ImageGenerator import Vector2D
from project_files.Config import config
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

    part_actual_rotations[curr_parent] = np.average(parent_rotations)
    for i, parent in enumerate(char.parents):
        rotation = part_actual_rotations[i]
        # if i not in char.parents:
        #     continue
        while parent is not None:
            rotation -= part_rotations[parent]
            parent = char.parents[parent]
        part_rotations[i] = rotation
    # part_rotations = part_actual_rotations
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
    # char_side_mirrored = project_files.ImageGenerator.char_side_mirrored
    folder = '\\'.join(filename.split('\\')[:-1])
    name = filename.split('\\')[-1].split('.')[0]
    test_inputs_folder = f"C:/School/Huji/Thesis/Pose_Estimation_in_2D_Characters/project_files/Test Inputs/{config['dataset']['character']}"
    im_file_names_annotated = []
    im_file_names = []
    with open(filename) as f:
        all_annotations = json.load(f)
        for i in range(len(all_annotations)):
            im_annotations = np.array([joint for joint in all_annotations[str(i)].values()]) * scale
            # im_annotations -= (scale - 1) * (char_front.image_size // 2)
            front_parameters = get_parameters_from_annotations(char_front, im_annotations)
            side_parameters = get_parameters_from_annotations(char_side, im_annotations)
            # side_parameters_mirrored = get_parameters_from_annotations(char_side_mirrored, im_annotations)
            # char = char_side_mirrored
            # parameters = side_parameters_mirrored
            if np.linalg.norm(front_parameters) < np.linalg.norm(side_parameters):
                char = char_front
                parameters = front_parameters
                # if np.linalg.norm(front_parameters) < np.linalg.norm(side_parameters_mirrored):
                #     char = char_front
                #     parameters = front_parameters
                # else:
                    # char = char_side_mirrored
                    # parameters = side_parameters_mirrored
            else:
                char = char_side
                parameters = side_parameters
                # if np.linalg.norm(side_parameters) < np.linalg.norm(side_parameters_mirrored):
                #     char = char_side
                #     parameters = side_parameters
                # else:
                    # char = char_side_mirrored
                    # parameters = side_parameters_mirrored

            im, data = project_files.ImageGenerator.create_image(char, parameters, as_image=False, random_order=False, random_generation=False)
            joints = np.array(data['joints'])
            joints = np.transpose(joints)

            test_im = Image.open(f"{test_inputs_folder}\\test_Pose{i+1}.png")
            test_im = test_im.resize((test_im.size[0] * scale, test_im.size[1] * scale))
            test_im = np.asarray(test_im)

            im = np.concatenate([test_im, im[:, :, :3]], axis=1)

            fig = plt.figure()
            plt.imshow(im)

            im_file_name = f"{folder}\\im{i}.png"
            im_file_names.append(im_file_name)
            plt.savefig(im_file_name)

            im_annotations = np.transpose(im_annotations)
            plt.scatter(im_annotations[0], im_annotations[1])
            plt.scatter(im_annotations[0] + char_front.image_size, im_annotations[1])

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
    GenerateSequence(r"C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\pose_estimation\output\Goofy\pose_resnet_16\angle_range_80\val\val_file\_iter_0_joints_pred.json")
