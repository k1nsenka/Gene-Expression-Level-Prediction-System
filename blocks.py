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
        with self.init_scope():
            pad = kernel // 2
            self.conv = nn.Conv1d(in_ch, out_ch*2, kernel, padding=pad, stride=stride)

    def forward(self, x):
        h = self.conv(x)
        h, g = F.split(h, 2, 1)
        h = F.dropout(h * F.sigmoid(g), self.do_rate)
        return h

class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilate, do_rate):
        super(DilatedBlock, self).__init__()
        self.do_rate = do_rate
        with self.init_scope():
            self.conv = nn.Conv1d(in_ch, out_ch*2, kernel, padding=dilate, dilation=dilate)

    def forward(self, xs):
        x = torch.cat(xs, axis=1)
        h = self.conv(x)
        h, g = torch.split(h, 2, 1)
        h = F.dropout(h * F.sigmoid(g), self.do_rate)
        return h