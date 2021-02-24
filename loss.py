from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2

        target = one_hot_embedding(y, 1 + self.num_classes)
        target = target[:, 1:]  # exclude background

        prob = x.sigmoid()
        pred = prob*target + (1-prob)*(1-target)         # pt = p if t > 0 else 1-p
        weight = alpha*target + (1-alpha)*(1-target)  # w = alpha if t > 0 else 1-alpha
        weight = weight * (1-pred).pow(gamma)
        weight = weight.detach()

        loss = F.binary_cross_entropy_with_logits(input=x, target=target, weight=weight, reduction='sum')
        return loss

    # def forward(self, loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets):
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # mask_loss = self.ce_loss(mask_preds, mask_targets)

        # return loc_loss, cls_loss, mask_loss, num_pos
        return loc_loss, cls_loss, num_pos
