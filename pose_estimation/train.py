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
import shutil
from datetime import datetime

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
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

    args = parser.parse_args()
    args.cfg = cfg

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    cfgs = \
        [
            # '2022-10-29_15-04-05-size=50000-angle_range=80-augmentations=False-num_frames=1',
            # '2022-10-29_15-29-21-size=50000-angle_range=80-augmentations=False-num_frames=2',
            # '2022-10-29_15-54-17-size=50000-angle_range=80-augmentations=False-num_frames=3',
            # '2022-10-29_16-19-16-size=50000-angle_range=80-augmentations=True-num_frames=3',
            # '2022-10-29_16-44-49-size=10000-angle_range=80-augmentations=True-num_frames=3',
            # '2022-10-29_16-49-58-size=100000-angle_range=80-augmentations=True-num_frames=3',
            # '2022-10-29_17-41-50-size=50000-angle_range=120-augmentations=True-num_frames=3',
            # '2022-10-29_18-07-24-size=50000-angle_range=40-augmentations=True-num_frames=3',
            # '2022-10-31_09-27-06-size=50000-angle_range=80-augmentations=True-num_frames=2',
            # '2022-10-31_09-52-44-size=50000-angle_range=120-augmentations=False-num_frames=2',
            # '2022-10-31_10-18-13-size=50000-angle_range=120-augmentations=True-num_frames=2',
            '2022-10-31_10-43-41-size=100000-angle_range=80-augmentations=False-num_frames=3',
            'size=50000-angle_range=80-augmentations=False-num_frames=2-layers=10',
            'size=50000-angle_range=120-augmentations=False-num_frames=2-layers=10',
            'size=100000-angle_range=80-augmentations=False-num_frames=3-layers=10',
            'size=50000-angle_range=120-augmentations=False-num_frames=2-layers=10-filters=64',
            'size=100000-angle_range=80-augmentations=False-num_frames=3-layers=10-filters=64',
        ]
    character = "Aang"
    for cfg in cfgs:
        path = f"C:\\School\\Huji\\Thesis\\Pose_Estimation_in_2D_Characters\\experiments\\{character}\\{cfg}.yaml"
        args = parse_args(path)
        reset_config(config, args)
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'train')
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, is_train=True
        )

        # copy model file
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
            final_output_dir)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                                 3,
                                 config.MODEL.IMAGE_SIZE[1],
                                 config.MODEL.IMAGE_SIZE[0]))
        writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

        gpus = [int(i) for i in config.GPUS.split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        # define loss function (criterion) and optimizer
        criterion = JointsMSELoss(
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT
        ).cuda()

        optimizer = get_optimizer(config, model)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
        )


        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TRAIN_SET,
            True,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(3, (3, 3)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                normalize,
            ])
        )
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST.BATCH_SIZE*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        best_perf = float('inf')
        best_model = False
        current_time = datetime.now()
        session = current_time.strftime("%S-%M-%H %d-%m-%Y")
        for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
            lr_scheduler.step()

            # train for one epoch
            train(config, train_loader, model, criterion, optimizer, epoch,
                  final_output_dir, tb_log_dir, writer_dict, logger)


            # evaluate on validation set
            perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                      criterion, final_output_dir, tb_log_dir,
                                      logger, writer_dict, val_file=session, prefix=f"epoch_{epoch}")
            logger.info(f"performance accuracy: {perf_indicator}")
            #TODO: maybe check this later
            if perf_indicator < best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
            # best_model = True

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


if __name__ == '__main__':
    main()
