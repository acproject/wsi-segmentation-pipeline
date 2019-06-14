

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import preprocessing


def lossfn(lossname, params=None):

    params = params if params is not None else {
        'reduction': 'none',
        'size_average': False,
        'ratio': 0.5,
        'scale_factor': 1/16,
        'gamma': 2,
        'alpha': [1/0.6670563,  1/0.05810939, 1/0.04750076, 1/0.22733354]       # see preprocessing.cls_ratios
    }

    params = preprocessing.DotDict(params)

    losses = {
        'xent': nn.CrossEntropyLoss(reduction=params.reduction, weight=torch.Tensor(params.alpha)),
        'focal': FocalLoss2d(gamma=params.gamma, alpha=params.alpha, size_average=params.size_average),
        'ohem': OHEM(ratio=params.ratio, scale_factor=params.scale_factor),
        'cent': ConditionalEntropyLoss(),
    }

    return losses[lossname]


####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.size_average = size_average
        self.alpha = alpha


        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)


    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -((1-pt)**self.gamma) * logpt

        if self.size_average:
            return loss.mean()

        return loss.view(-1, input.size(0))


class OHEM(torch.nn.NLLLoss):
    """ Online hard example mining."""

    def __init__(self, ratio, scale_factor=0.125):
        super(OHEM, self).__init__(None, True)
        self.ratio = ratio
        self.scale_factor = scale_factor

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio

        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        y = torch.nn.functional.interpolate(y.unsqueeze_(0).float(), mode='nearest', scale_factor=self.scale_factor).long().squeeze_(0)

        num_inst = x.size(0)

        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label].mean()
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        if x_hn.size(0) == 0:
            return torch.nn.functional.cross_entropy(x, y) * 0
        return torch.nn.functional.cross_entropy(x_hn, y_hn)


class ConditionalEntropyLoss(torch.nn.Module):
    """ conditional entropy + cross entropy combined ."""

    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x, y):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b + F.cross_entropy(x, y, reduction='none')

