# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate, validate_og
from utils.utils import create_logger

import dataset
import models


def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    args.cfg = cfg
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()
    args.cfg = cfg

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main():
    cfgs = \
        [
            '2022-10-29_15-04-05-size=50000-angle_range=80-augmentations=False-num_frames=1',
            '2022-10-29_15-29-21-size=50000-angle_range=80-augmentations=False-num_frames=2',
            '2022-10-29_15-54-17-size=50000-angle_range=80-augmentations=False-num_frames=3',
            '2022-10-29_16-19-16-size=50000-angle_range=80-augmentations=True-num_frames=3',
            '2022-10-29_16-44-49-size=10000-angle_range=80-augmentations=True-num_frames=3',
            '2022-10-29_16-49-58-size=100000-angle_range=80-augmentations=True-num_frames=3',
            '2022-10-29_17-41-50-size=50000-angle_range=120-augmentations=True-num_frames=3',
            '2022-10-29_18-07-24-size=50000-angle_range=40-augmentations=True-num_frames=3',
            '2022-10-31_09-27-06-size=50000-angle_range=80-augmentations=True-num_frames=2',
            '2022-10-31_09-52-44-size=50000-angle_range=120-augmentations=False-num_frames=2',
            '2022-10-31_10-18-13-size=50000-angle_range=120-augmentations=True-num_frames=2',
            '2022-10-31_10-43-41-size=100000-angle_range=80-augmentations=False-num_frames=3',
            'size=50000-angle_range=80-augmentations=False-num_frames=2-layers=10',
            'size=50000-angle_range=120-augmentations=False-num_frames=2-layers=10',
            'size=100000-angle_range=80-augmentations=False-num_frames=3-layers=10',
            'size=50000-angle_range=120-augmentations=False-num_frames=2-layers=10-filters=64',
            'size=100000-angle_range=80-augmentations=False-num_frames=3-layers=10-filters=64',
        ]
    character = "Aang"
    for cfg in cfgs:
        no_paf = True
        if no_paf:
            cfg = cfg + "(no-paf)"
            path = f"C:\\School\\Huji\\Thesis\\Pose_Estimation_in_2D_Characters\\experiments\\{character}\\no_paf\\{cfg}.yaml"
        else:
            path = f"C:\\School\\Huji\\Thesis\\Pose_Estimation_in_2D_Characters\\experiments\\{character}\\{cfg}.yaml"
        args = parse_args(path)
        reset_config(config, args)
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'valid')
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=False
        )

        if config.TEST.MODEL_FILE:
            logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            state_dict = torch.load(config.TEST.MODEL_FILE)
            new_state_dict = {'.'.join(k.split('.')[1:]): state_dict[k] for k in state_dict}
            model.load_state_dict(new_state_dict)
        else:
            model_state_file = os.path.join(final_output_dir,
                                            'final_state.pth.tar')
            logger.info('=> loading model from {}'.format(model_state_file))
            new_state_dict = torch.load(model_state_file)
            # new_state_dict = {'.'.join(k.split('.')[1:]): new_state_dict[k] for k in new_state_dict}
            model.load_state_dict(new_state_dict)

        gpus = [int(i) for i in config.GPUS.split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        # define loss function (criterion) and optimizer
        criterion = JointsMSELoss(
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT
        ).cuda()

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TEST_SET,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(3, (3, 3)),
                normalize,
            ])
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST.BATCH_SIZE*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model, criterion,
                 final_output_dir, tb_log_dir, logger, og=no_paf)
        logger.info(f"performance accuracy: {perf_indicator}")
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def main_og():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate_og(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir, logger)


if __name__ == '__main__':
    main()
