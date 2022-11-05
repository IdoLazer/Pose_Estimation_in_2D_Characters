# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import matplotlib.pyplot as plt
from utils.transforms import transform_preds
# from transforms import transform_preds


def detect_peaks(image, num_peaks):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image==0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    idxs = np.where(detected_peaks)
    maxvals = image[idxs]
    maxval = np.max(maxvals)
    idxs = np.transpose(np.stack([idxs[0], idxs[1]]))
    idxs = idxs[maxvals > np.min((0.3, maxval / 2))]
    maxvals = maxvals[maxvals > np.min((0.3, maxval / 2))]
    idxs = [idx for val, idx in sorted(zip(maxvals, idxs), key=lambda pair: pair[0])]
    if len(idxs) > num_peaks:
        idxs = idxs[-num_peaks:]

    return idxs, maxvals


def calc_edge_score(q1, q2, paf):
    length = np.linalg.norm(q2 - q1)
    n = np.max((int(length) // 2, 1))
    x, y = np.rint(np.linspace(q1, q2, n)).astype(int).transpose()
    normalized = (q2 - q1) / np.max((length, np.finfo(float).eps))
    return np.sum(paf[(x, y)] * normalized)


class JointNode:

    def __init__(self, id, pos, val):
        self.id = id
        self.pos = pos
        self.val = val
        self.parent = None
        self.children_candidates = {}
        self.children = []
        self.score = None

    def add_child_candidate(self, id, child, edge_score):
        if id in self.children_candidates.keys():
            self.children_candidates[id].append((child, edge_score))
        else:
            self.children_candidates[id] = [(child, edge_score)]
        child.parent = self

    def calc_score(self):
        if len(self.children_candidates) > 0:
            ids = sorted(self.children_candidates)
            for id in ids:
                child_score = -1
                max_child = None
                child_candidates = self.children_candidates[id]
                for child_candidate, edge_score in child_candidates:
                    if child_candidate.score is None:
                        child_candidate.calc_score()
                    if max_child is None or child_candidate.score + edge_score > child_score:
                        child_score = child_candidate.score + edge_score
                        max_child = child_candidate
                if self.score is None:
                    self.score = child_score
                else:
                    self.score += child_score
                self.children.append(max_child)
        else:
            self.score = 0


def get_max_node(nodes):
    score = None
    max_node = None
    for node in nodes:
        node.calc_score()
        if score is None or node.score > score:
            score = node.score
            max_node = node
    return max_node


def get_preds_from_tree(curr_node, preds):
    preds[curr_node.id] = ([curr_node.pos[1], curr_node.pos[0]], [curr_node.val])
    if len(curr_node.children) > 0:
        for child in curr_node.children:
            get_preds_from_tree(child, preds)


def get_max_preds_with_pafs(batch_heatmaps, batch_pafs, n, limbs):
    preds = []
    maxvals = []
    for j, (heatmap, pafs) in enumerate(zip(batch_heatmaps, batch_pafs)):
        nodes = {}
        for i, jointmap in enumerate(heatmap):
            joint_peaks, joint_maxvals = detect_peaks(jointmap, n)
            nodes[i] = [JointNode(i, peak, val) for peak, val in zip(joint_peaks, joint_maxvals)]
        for limb_index, limb in enumerate(limbs):
            paf = pafs[limb_index]
            parent_joint = limb[0]
            child_joint = limb[1]
            for node in nodes[parent_joint]:
                for child in nodes[child_joint]:
                    node.add_child_candidate(child_joint, child, calc_edge_score(node.pos, child.pos, paf))
        root_node = get_max_node(nodes[0])
        sample_preds = {}
        get_preds_from_tree(root_node, sample_preds)
        preds.append(np.array([sample_preds[pred_id][0] for pred_id in sorted(sample_preds)]))
        maxvals.append(np.array([sample_preds[pred_id][1] for pred_id in sorted(sample_preds)]))
    preds = np.array(preds)
    maxvals = np.array(maxvals)
    return preds.astype(np.float32), maxvals.astype(np.float32)





def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, batch_pafs, limbs, center, scale):
    # coords, maxvals = get_max_preds(batch_heatmaps)
    coords, maxvals = get_max_preds_with_pafs(batch_heatmaps, batch_pafs, 3, limbs)
    coords_original = coords.copy()
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals, coords_original


def get_final_preds_og(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    coords_original = coords.copy()
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals, coords_original
