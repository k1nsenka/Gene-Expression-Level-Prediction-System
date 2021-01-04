import torch
#import chainer.links as L
import torch.nn as nn
#import chainer.functions as F
import torch.nn.functional as F
import numpy as np


class SqueezeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, do_rate):
        super(SqueezeBlock, self).__init__()
        self.do_rate = do_rate
        pad = kernel // 2
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch*2, kernel, padding=pad, stride=stride), name='weight')

    def forward(self, x):
        #print(x.shape)
        h = self.conv(x)
        h, g = torch.chunk(h, 2, dim=1)
        h = F.dropout(h * torch.sigmoid(g), self.do_rate)
        return h

class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilate, do_rate):
        super(DilatedBlock, self).__init__()
        self.do_rate = do_rate
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch*2, kernel, padding=dilate, dilation=dilate), name='weight')

    def forward(self, xs):
        #print(len(xs))
        x = torch.cat(xs, dim=1)
        #print(x.shape)
        h = self.conv(x)
        h, g = torch.chunk(h, 2, 1)
        h = F.dropout(h * torch.sigmoid(g), self.do_rate)
        return h