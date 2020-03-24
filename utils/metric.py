from __future__ import print_function, absolute_import
import torch
from torch import nn

__all__ = ['KLDivLoss', 'distillation']

def KLDivLoss(logits_q, logits_p, T):
    assert logits_p.size() == logits_q.size()
    b, c = logits_p.size()
    p = nn.Softmax(dim=1)(logits_p / T)
    q = nn.Softmax(dim=1)(logits_q / T)
    epsilon = 1e-8
    _p = (p + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    _q = (q + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    return (T**2) * torch.mean(torch.sum(_p * torch.log(_p / _q), dim=1))


def distillation(y, teacher_scores, T):
    q = nn.LogSoftmax(dim=1)(y / T)
    p = nn.Softmax(dim=1)(teacher_scores / T)
    return  (T**2) * nn.KLDivLoss(reduction='batchmean')(q, p)

