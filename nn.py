import torch
#import chainer.links as L
import torch.nn as nn
#import chainer.functions as F
import torch.nn.functional as f


import blocks


bc = 24 # base channel

default_squeeze_params = [
    # out_ch, kernel, stride, dropout
    [bc*2, 21, 2, 0], #1 128 -> 64
    [int(bc*2.5), 7, 4, 0.05], #2  64 -> 16
    [int(bc*3.2), 7, 4, 0.05], #3  16 -> 4
    [bc*4, 7, 4, 0.05]  #4  4 -> 1
]


default_dilated_params = [
# out_ch, kernel, dilated, dropout
    [bc, 3, 1, 0.1],
    [bc, 3, 2, 0.1],
    [bc, 3, 4, 0.1],
    [bc, 3, 8, 0.1],
    [bc, 3, 16, 0.1],
    [bc, 3, 32, 0.1],
    [bc, 3, 64, 0.1]
]


class Net(nn.Module):
    def __init__(self, squeeze_params=default_squeeze_params, dilated_params=default_dilated_params, n_targets=10):
        super(Net, self).__init__()
        self._n_squeeze = len(squeeze_params)
        self._n_dilated = len(dilated_params)
        with self.init_scope():
            in_ch = 4
            for i, param in enumerate(squeeze_params):
                out_ch, kernel, stride, do_rate = param
                setattr(self, "s_{}".format(i), SqueezeBlock(in_ch, out_ch, kernel, stride, do_rate))
                in_ch = out_ch
            for i, param in enumerate(dilated_params):
                out_ch, kernel, dilated, do_rate = param
                setattr(self, "d_{}".format(i), DilatedBlock(in_ch, out_ch, kernel, dilated, do_rate))
                in_ch += out_ch
            self.l = nn.Conv1d(in_ch, n_targets, 1)

    def forward(self, x):
        # x : (B, X, 4)
        xp = torch.Tensor(x)
        h = xp.numpy().transpose(0, 2, 1)
        h = h.astype(xp.float32)

        for i in range(self._n_squeeze):
            h = self["s_{}".format(i)](h)

        hs = [h]
        for i in range(self._n_dilated):
            h = self["d_{}".format(i)](hs)
            hs.append(h)

        h = self.l(torch.cat(hs, axis=1))
        h = xp.numpy().transpose(0, 2, 1)
        return h