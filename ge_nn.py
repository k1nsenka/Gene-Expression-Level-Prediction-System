import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


import ge_blocks


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
    def __init__(self, squeeze_params=default_squeeze_params, dilated_params=default_dilated_params, n_targets=4229):
        super(Net, self).__init__()
        self._n_squeeze = len(squeeze_params)
        self._n_dilated = len(dilated_params)
        #squeez
        #Squeez Block 0
        in_ch = 4
        out_ch = bc*2
        self.s_0 = ge_blocks.SqueezeBlock(in_ch, out_ch, 21, 2, 0)
        in_ch = out_ch
        #Squeez Block 1
        out_ch = int(bc*2.5)
        self.s_1 = ge_blocks.SqueezeBlock(in_ch, out_ch, 7, 4, 0.05)
        in_ch = out_ch
        #Squeez Block 2
        out_ch = int(bc*3.2)
        self.s_2 = ge_blocks.SqueezeBlock(in_ch, out_ch, 7, 4, 0.05)
        in_ch = out_ch
        #Squeez Block 3
        out_ch = bc*4
        self.s_3 = ge_blocks.SqueezeBlock(in_ch, out_ch, 7, 4, 0.05)
        in_ch = out_ch

        #dilated
        out_ch = bc
        #Dilated Block 0
        self.d_0 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 1, 0.1)
        in_ch += out_ch
        #Dilated Block 1
        self.d_1 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 2, 0.1)
        in_ch += out_ch
        #Dilated Block 2
        self.d_2 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 4, 0.1)
        in_ch += out_ch
        #Dilated Block 3
        self.d_3 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 8, 0.1)
        in_ch += out_ch
        #Dilated Block 4
        self.d_4 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 16, 0.1)
        in_ch += out_ch
        #Dilated Block 5
        self.d_5 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 32, 0.1)
        in_ch += out_ch
        #Dilated Block 6
        self.d_6 = ge_blocks.DilatedBlock(in_ch, out_ch, 3, 64, 0.1)
        in_ch += out_ch

        self.l = nn.Conv1d(264, n_targets, 1)

    def forward(self, x):
        # x : (B, X, 4)
        xp = torch.Tensor(x)
        h = xp.transpose(2, 1)
        #squeez
        h1 = self.s_0(h)
        h2 = self.s_1(h1)
        h3 = self.s_2(h2)
        h4 = self.s_3(h3)
        #dilated
        hs = [h4]
        hs0 = self.d_0(hs)
        hs.append(hs0)
        hs1 = self.d_1(hs)
        hs.append(hs1)
        hs2 = self.d_2(hs)
        hs.append(hs2)
        hs3 = self.d_3(hs)
        hs.append(hs3)
        hs4 = self.d_4(hs)
        hs.append(hs4)
        hs5 = self.d_5(hs)
        hs.append(hs5)
        hs6 = self.d_6(hs)
        hs.append(hs6)
        #last
        hsl = torch.cat(hs, dim=1)
        print(hsl.shape)

        h = self.l(hsl)
        h = h.transpose(2, 1)
        return h