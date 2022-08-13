# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import json_tricks as json
import project_files.SequenceGenerator as sequence
from core.inference import get_max_preds, get_max_preds_with_pafs


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, parents, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            for joint in joints:
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
            for joint_idx, parent_idx in enumerate(parents):
                joint = joints[joint_idx]
                joint_vis = joints_vis[joint_idx]
                if joint_vis[0]:
                    if parent_idx >= 0:
                        joint_parent = joints[parent_idx]
                        joint_parent_vis = joints_vis[parent_idx]
                        if joint_parent_vis[0]:
                            cv2.line(ndarr, (int(joint[0]), int(joint[1])),
                                     (int(joint_parent[0]), int(joint_parent[1])),
                                     [255, 255, 0], 1)
            k = k + 1
    # k = 0
    # for y in range(ymaps):
    #     for x in range(xmaps):
    #         if k >= nmaps:
    #             break
    #         joints = batch_joints[k]
    #         joints_vis = batch_joints_vis[k]
    #
    #         for joint, joint_vis in zip(joints, joints_vis):
    #             joint[0] = x * width + padding + joint[0]
    #             joint[1] = y * height + padding + joint[1]
    #             if joint_vis[0]:
    #                 cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [255, 0, 0], 2)
    #         k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_pafs(batch_image, batch_pafs, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_pafs: ['batch_size, num_limbs, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_pafs.size(0)
    num_limbs = batch_pafs.size(1)
    paf_height = batch_pafs.size(2)
    paf_width = batch_pafs.size(3)

    grid_image = np.zeros((batch_size*paf_height,
                           (num_limbs+1)*paf_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        pafs = batch_pafs[i].abs()\
                            .mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(paf_width), int(paf_height)))

        height_begin = paf_height * i
        height_end = paf_height * (i + 1)
        for j in range(num_limbs):
            paf = np.zeros((paf_width, paf_height, 3))
            paf[:, :, :2] = pafs[j, :, :, :]
            masked_image = paf*0.7 + resized_image*0.3

            width_begin = paf_width * (j+1)
            width_end = paf_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:paf_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, hm_target, paf_target, joints_pred, hm_output, paf_output,
                      prefix, vis_sequence=False):
    if not config.DEBUG.DEBUG:
        return

    joints_dict = {}
    for i, joints in enumerate(joints_pred):
        joints_dict[i] = {}
        for j, joint in enumerate(joints):
            joints_dict[i][j] = joint.tolist()
    json.dump(joints_dict, '{}_joints_pred.json'.format(prefix))

    if vis_sequence:
        sequence.GenerateSequence('{}_joints_pred.json'.format(prefix))

    joints_dict = {}
    for i, joints in enumerate(meta['joints']):
        joints_dict[i] = {}
        for j, joint in enumerate(joints):
            joints_dict[i][j] = joint.tolist()
    json.dump(joints_dict, '{}_joints_meta.json'.format(prefix))

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix),
            config.MODEL.PARENTS
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix),
            config.MODEL.PARENTS
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, hm_target, '{}_hm_gt.jpg'.format(prefix)
        )
        save_batch_pafs(
            input, paf_target, '{}_paf_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, hm_output, '{}_hm_pred.jpg'.format(prefix)
        )
        save_batch_pafs(
            input, paf_output, '{}_paf_pred.jpg'.format(prefix)
        )


# from dataset.JointsDataset import JointsDataset
# import torch
# import matplotlib.pyplot as plt
# from matplotlib import image
# if __name__ == "__main__":
#     img1 = cv2.imread(
#         r'C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\data\aang\images\pose80000.png',
#         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(float) / 255.0
#     img2 = cv2.imread(
#         r'C:\School\Huji\Thesis\Pose_Estimation_in_2D_Characters\data\aang\images\pose80001.png',
#         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(float) / 255.0
#     img1 = img1.transpose((2, 0, 1))
#     img2 = img2.transpose((2, 0, 1))
#     img1 = np.expand_dims(img1, axis=0)
#     img2 = np.expand_dims(img2, axis=0)
#     img1 = torch.tensor(img1)
#     img2 = torch.tensor(img2)
#
#
#     img = img1
#
#
#     joints_vis = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     joints1 = [[66.0, 55.0], [67.03310121575242, 36.040513783553166], [55.03328398429531, 42.08383461918481],
#               [71.41034881665577, 41.22554902014642], [61.471696577536335, 67.0137846724653],
#               [68.61309588967151, 65.6919275728259], [68.4979720127729, 55.23753961088386],
#               [87.2980156973691, 37.38773474424357], [82.20104433501098, 53.59155926895511],
#               [41.26269838952197, 71.74859745560875], [71.6779779373538, 65.4729474081253],
#               [85.34910096630446, 48.5874205773887], [85.40100226307035, 67.51770064097218],
#               [28.473758677319395, 70.59584587871416]]
#
#     joints2 = [[67.0, 56.0], [71.48911221993933, 30.879327933405918], [57.47207697774843, 38.11238942414819],
#                [74.34622995391518, 37.996726308911406], [61.058620735826636, 70.57154408743857],
#                [68.38642810188917, 69.90317041922478], [39.66144036301456, 29.740822197357517],
#                [80.9470822542872, 55.206675656320684], [45.28221833009934, 94.51577996438365],
#                [33.0871979685606, 59.8051871734158], [28.19449049823703, 24.393695659732316],
#                [84.01094001885161, 43.91821335113994], [55.976302757657685, 102.48466277483678],
#                [31.404406331047106, 46.00753211044598]]
#
#     joints1 = np.array(joints1)
#     joints2 = np.array(joints2)
#     joints_vis = np.array(joints_vis)
#     joints_3d_vis = np.zeros((len(joints1), 3), dtype=float)
#     joints_3d_vis[:, 0] = joints_vis[:]
#     joints_3d_vis[:, 1] = joints_vis[:]
#
#     heatmaps1, _ = JointsDataset.generate_target(joints1, joints_3d_vis, np.array([128, 128]), np.array([128, 128]), 2)
#     heatmaps2, _ = JointsDataset.generate_target(joints2, joints_3d_vis, np.array([128, 128]), np.array([128, 128]), 2)
#     heatmaps1 = np.expand_dims(heatmaps1, axis=0)
#     heatmaps2 = np.expand_dims(heatmaps2, axis=0)
#
#
#     heatmaps = (heatmaps1 + heatmaps2) / 2
#
#
#     save_batch_heatmaps(
#         img, torch.tensor(heatmaps), r'C:\School\Huji\Thesis\crap\test_heatmaps.jpg'
#     )
#
#     parent_ids = [-1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9]
#     limbs = []
#     for i, parent_id in enumerate(parent_ids):
#         if parent_id >= 0:
#             limbs.append([parent_id, i])
#     pafs1, _ = JointsDataset.generate_paf(joints1, limbs, 5, np.array([128, 128]), np.array([128, 128]))
#     pafs2, _ = JointsDataset.generate_paf(joints2, limbs, 5, np.array([128, 128]), np.array([128, 128]))
#     pafs1 = np.expand_dims(pafs1, axis=0)
#     pafs2 = np.expand_dims(pafs2, axis=0)
#
#
#     pafs = pafs1
#
#
#     save_batch_pafs(
#         img, torch.tensor(pafs), r'C:\School\Huji\Thesis\crap\test_pafs.jpg'
#     )
#
#     preds, maxvals = get_max_preds_with_pafs(heatmaps, pafs, 3, limbs)
#     save_batch_image_with_joints(
#         img, preds, np.expand_dims(joints_3d_vis, axis=0),
#         r'C:\School\Huji\Thesis\crap\test3.jpg',
#         parent_ids
#     )