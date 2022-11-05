import json
import numpy as np
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt


def get_gt_joints(char):
    with open(f"Results/{char}/gt/gt.json") as f:
        gt = json.load(f)
    return [entry['joints'] for entry in gt]


def get_our_joints(char):
    with open(f"Results/{char}/Ours/_iter_0_joints_pred.json") as f:
        ours = json.load(f)

    centers = []
    with open(f"Results/{char}/Baseline/test.json") as f:
        test_annot = json.load(f)
        for image in test_annot:
            centers.append(image['center'])

    return [[entry[str(joint)] for joint in range(len(entry))] for entry in
            [(ours[str(idx)]) for idx in range(len(ours))]]


def get_baseline_joints(char):
    joints_order = [6, 7, 8, 12, 13, 2, 3, 11, 14, 1, 4, 10, 15, 0, 5]
    with open(f"Results/{char}/Baseline/val_0_joints_pred.json") as f:
        baseline = json.load(f)
    centers = []
    with open(f"Results/{char}/Baseline/test.json") as f:
        test_annot = json.load(f)
        for image in test_annot:
            centers.append([image['center']] * (len(joints_order) - 1))
    joints = [[entry[str(joint)] for joint in joints_order] for entry in
              [(baseline[str(idx)]) for idx in range(len(baseline))]]
    joints = np.array(joints)
    roots = joints[:, 0, :] * 0.75 + joints[:, 1, :] * 0.25
    joints = joints[:, 1:, :]
    joints[:, 0, :] = roots
    joints -= np.array([128, 128])
    joints *= 0.78
    joints += np.array(centers)
    return list(joints)


def get_efficient_post_joints(char, im_size):
    root_joints_names = ['pelvis', 'thorax']
    joints_names = ['upper_neck', 'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip', 'right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_wrist', 'left_wrist', 'right_ankle', 'left_ankle']

    # find all joint files and save them according to the pose frame number
    files = {}
    for file in os.listdir(f"Results/{char}/EfficientPose"):
        if file.endswith(".csv"):
            files[file.split('_')[1][4:]] = f"Results/{char}/EfficientPose/" + file

    # extract all joint positions from csv files in the correct order
    all_joint_positions = []
    for i in range(len(files)):
        with open(files[str(i + 1)]) as csv_file:
            csv_reader = csv.reader(csv_file)
            lines = []
            joints = {}
            for line in csv_reader:
                if len(line) > 0:
                    lines.append(line)
            for j in range(len(lines[0])):
                joints[lines[0][j]] = float(lines[1][j])
            all_joint_positions.append(joints)

    # convert joint positions from efficient pose format to our format, including omitting some joints and interpolating
    # others
    final_joint_positions = []
    for joints in all_joint_positions:
        joint_positions = []
        root_1 = np.array([joints[f"{root_joints_names[0]}_x"], joints[f"{root_joints_names[0]}_y"]])
        root_2 = np.array([joints[f"{root_joints_names[1]}_x"], joints[f"{root_joints_names[1]}_y"]])
        root = 0.75 * root_1 + 0.25 * root_2
        joint_positions.append(list(root * im_size))
        for joint_name in joints_names:
            joint_positions.append(list(im_size * np.array([joints[f"{joint_name}_x"], joints[f"{joint_name}_y"]])))
        final_joint_positions.append(joint_positions)
    return final_joint_positions


def get_openpose_joints(char, im_size):
    joints_order = [8, 1, 0, 2, 5, 12, 9, 3, 6, 13, 10, 4, 7, 14, 11]

    # find all joint files and save them according to the pose frame number
    files = {}
    for file in os.listdir(f"Results/{char}/OpenPose"):
        if file.endswith(".json"):
            files[file.split('_')[1][4:]] = f"Results/{char}/OpenPose/" + file

    # extract all joint positions from csv files in the correct order
    all_joint_positions = []
    for i in range(len(files)):
        with open(files[str(i + 1)]) as f:
            keypoints = json.load(f)
            keypoints = keypoints['people']
        if len(keypoints) > 0:
            keypoints = keypoints[0]
            keypoints = keypoints['pose_keypoints_2d']
            joints_positions = [[keypoints[i * 3], keypoints[i * 3 + 1]] for i in range(len(keypoints) // 3)]
        else:
            joints_positions = [[im_size // 2, im_size // 2]] * 25
        joints_positions = [joints_positions[i] for i in joints_order]
        all_joint_positions.append(joints_positions)

    all_joint_positions = np.array(all_joint_positions)
    root_joints = all_joint_positions[:, 0] * 0.75 + all_joint_positions[:, 1] * 0.25
    head_joints = all_joint_positions[:, 1] * 0.75 + all_joint_positions[:, 2] * 0.25
    joints = all_joint_positions[:, 1:]
    joints[:, 0] = root_joints
    joints[:, 1] = head_joints

    return list(joints)


def calculate_performance(gt, pred, im_size):
    gt = np.array(gt)
    pred = np.array(pred)
    d = np.linalg.norm(gt - pred, axis=2)
    return np.average(np.sum(d, axis=1) / im_size)


def visualize_joints(char, im_size, joints, method_names):
    parents = [None, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9]
    files = {}
    for file in os.listdir(f"Results/{char}/Images/inputs"):
        if file.endswith(".png"):
            files[file.split('.')[0].split('_')[1][4:]] = f"Results/{char}/Images/inputs/" + file

    for i in range(len(files)):
        im = Image.open(files[str(i + 1)])
        fig, axes = plt.subplots(1, len(joints), figsize=((im_size / 75) * len(joints), im_size / 75))
        for j, ax in enumerate(axes):
            ax.imshow(im)
            ax.set_title(method_names[j])
            for k, parent in enumerate(parents):
                if parent is not None:
                    ax.plot([joints[j][i][k][0], joints[j][i][parent][0]], [joints[j][i][k][1], joints[j][i][parent][1]]
                            , marker='o', ms=1.5, mfc='white', mec='white', linewidth=2)
            ax.axis('off')
        fig.savefig(f"Results/{char}/Images/outputs/{i + 1}.png")
        plt.close(fig)


def main():
    char = "Aang"
    im_size = 128
    gt = get_gt_joints(char)
    ours = get_our_joints(char)
    baseline_joints = get_baseline_joints(char)
    efficient_pose_joints = get_efficient_post_joints(char, im_size)
    openpose_joints = get_openpose_joints(char, im_size)
    our_score = calculate_performance(gt, ours, im_size)
    baseline_score = calculate_performance(gt, baseline_joints, im_size)
    efficient_pose_score = calculate_performance(gt, efficient_pose_joints, im_size)
    openpose_score = calculate_performance(gt, openpose_joints, im_size)
    visualize_joints(char, 256, [gt, ours, baseline_joints, efficient_pose_joints, openpose_joints],
                     ["Ground Truth", "Ours", "Baselines", "EfficientPose", "OpenPose"])
    print(f"Scores:\n"
          f"ours={our_score}\n"
          f"baseline={baseline_score}\n"
          f"efficient_pose={efficient_pose_score}\n"
          f"openpose={openpose_score}")


if __name__ == "__main__":
    main()
