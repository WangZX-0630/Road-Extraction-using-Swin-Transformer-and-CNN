import torch
import torch.nn as nn
import cv2
import numpy as np
from models.utils import resize
from models.builder import build_loss
from models.losses import accuracy
from mmcv.runner import BaseModule


class Losses(object):
    def __init__(self,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255):
        self.ignore_index = ignore_index
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        self.align_corners = False

    def build_losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        # seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=None,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=None,
                    ignore_index=self.ignore_index)

        # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def count_total_loss(self, losses):
        total_loss = 0
        for name, loss in losses.items():
            total_loss += loss
        return total_loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


if __name__ == '__main__':
    torch.manual_seed(1234)
    a = torch.rand(4, 2, 10, 10).cuda()
    b = torch.rand(4, 10, 10).cuda().long()
    c = torch.rand(4, 10, 10).cuda()
    d = torch.rand(4, 10, 10).cuda()
    criterion = Losses(loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0), dict(type='DiceLoss', loss_weight=1.5)])
    print(criterion.count_total_loss(criterion.build_losses(a, b)))
    loss = dice_bce_loss()
    print(loss(d, c))
