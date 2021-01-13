import torch
import numpy as np
import h5py
import ignite.contrib.metrics.regression as regression
from sklearn.metrics import r2_score
import sys
import torch.nn as nn
import matplotlib.pyplot as plt

import ge_nn



#テストデータ
#data = h5py.File('/Users/nemomac/gelp/dataset/l131k_w128.h5', 'r')
data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
#(batchsize, 131072, 4)
test_in = data['test_in']


x = test_in[0]
#x = torch.Tensor(x)
print(len(x))
seq = []

for i in range(len(x)):
    for base in range(4):
        if x[i, base] == True:
            if base == 0:
                seq.append('A')
                break
            elif base == 1:
                seq.append('C')
                break
            elif base == 2:
                seq.append('G')
                break
            elif base == 3:
                seq.append('T')
                break

with open('seq.txt', 'w') as f:
    for i in range(len(seq)):
        f.write(seq[i])