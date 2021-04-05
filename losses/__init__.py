import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MLHingeLoss(nn.Module):

    def forward(self, x, y, reduction='mean'):
        """
            y: labels have standard {0,1} form and will be converted to indices
        """
        b, c = x.size()
        idx = (torch.arange(c) + 1).type_as(x)
        y_idx, _ = (idx * y).sort(-1, descending=True)
        y_idx = (y_idx - 1).long()

        return F.multilabel_margin_loss(x, y_idx, reduction=reduction)

class CrossEntropy(nn.Module):

    def forward(self, x, y):
        return F.cross_entropy(x, torch.argmax(y, -1))

class Separate(nn.Module):

    def __init__(self):
        super().__init__()
        total = 8675
        class_size = torch.tensor([1352, 874, 1050, 629, 1063, 932, 1029, 957, 956, 949.])
        pw = (total - class_size) / class_size
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x, y):
        return self.loss(x, y)

def get_criterion(loss_name, **kwargs):

    losses = {
            "SoftMargin": nn.MultiLabelSoftMarginLoss,
            "Hinge": MLHingeLoss,
            "CrossEntropy": CrossEntropy,
            "Separate": Separate
            }

    return losses[loss_name](**kwargs)


#
# Mask self-supervision
#
def mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    weight = pseudo_gt.sum(1).type_as(mask_gt)
    mask_gt += (1 - weight) * ignore_index

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index)
    return loss.mean()
