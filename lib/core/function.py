# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
from pathlib import Path

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds_with_pafs
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, hm_target, hm_target_weight, paf_target, paf_target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        # outputs = model(input)
        output = model(input)
        # loss = 0
        # for output in outputs:
        hm_output, paf_output = output[:, :config.MODEL.NUM_JOINTS, :, :], output[:, config.MODEL.NUM_JOINTS:, :, :]
        paf_output = torch.stack((paf_output[:, :config.MODEL.NUM_LIMBS, :, :],
                                  paf_output[:, config.MODEL.NUM_LIMBS:, :, :]), axis=3).mT
        hm_target = hm_target.cuda(non_blocking=True)
        hm_target_weight = hm_target_weight.cuda(non_blocking=True)
        paf_target = paf_target.cuda(non_blocking=True)
        paf_target_weight = paf_target_weight.cuda(non_blocking=True)

        # loss += criterion(hm_output, hm_target, hm_target_weight)
        # loss += criterion(paf_output, paf_target, paf_target_weight)
        loss = criterion(hm_output, hm_target, hm_target_weight) + criterion(paf_output, paf_target, paf_target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(hm_output.detach().cpu().numpy(),
                                         # paf_output.detach().cpu().numpy(),
                                         hm_target.detach().cpu().numpy(),
                                         # paf_target.detach().cpu().numpy(),
                                         # config.MODEL.LIMBS
                                         )
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, hm_target, paf_target, pred*2, hm_output, paf_output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, val_file='val_file', prefix=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, hm_target, hm_target_weight, paf_target, paf_target_weight, meta) in enumerate(val_loader):
            # compute output
            # outputs = model(input)[1]
            output = model(input)
            hm_output, paf_output = output[:, :config.MODEL.NUM_JOINTS, :, :], output[:, config.MODEL.NUM_JOINTS:, :, :]
            paf_output = torch.stack((paf_output[:, :config.MODEL.NUM_LIMBS, :, :],
                                      paf_output[:, config.MODEL.NUM_LIMBS:, :, :]), axis=3).mT
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)
                hm_output_flipped, paf_output_flipped = outputs_flipped[:, :config.MODEL.NUM_JOINTS, :, :],\
                                                        outputs_flipped[:, config.MODEL.NUM_JOINTS:, :, :]
                hm_output_flipped = flip_back(hm_output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                hm_output_flipped = torch.from_numpy(hm_output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    hm_output_flipped[:, :, :, 1:] = \
                        hm_output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                hm_output = (hm_output + hm_output_flipped) * 0.5

            hm_target = hm_target.cuda(non_blocking=True)
            hm_target_weight = hm_target_weight.cuda(non_blocking=True)
            paf_target = paf_target.cuda(non_blocking=True)
            paf_target_weight = paf_target_weight.cuda(non_blocking=True)

            loss = criterion(hm_output, hm_target, hm_target_weight) + criterion(paf_output, paf_target,
                                                                                 paf_target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(hm_output.cpu().numpy(),
                                             hm_target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, hm_output.clone().cpu().numpy(), paf_output.clone().cpu().numpy(), config.MODEL.LIMBS, c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)
                val_file = os.path.join(output_dir, 'val', val_file)
                val_path = Path(val_file)
                if not val_path.exists():
                    print('=> creating {}'.format(val_path))
                    val_path.mkdir(parents=True)
                prefix = f"{os.path.join(val_path, prefix)}_iter_{i}"
                vis_preds, _ = get_max_preds_with_pafs(hm_output.clone().cpu().numpy(),
                                                       paf_output.clone().cpu().numpy(),
                                                       3,
                                                       config.MODEL.LIMBS)
                save_debug_images(config, input, meta, hm_target, paf_target, vis_preds * 2, hm_output, paf_output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
