import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py


import ge_data
import ge_loss
import ge_nn

print('opening file ...')
ml_h5 = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
#ml_h5 = h5py.File('/home/abe/data/genome_data/seq.h5')
#ml_h5 = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')

print('calling dataloader ...')
max_shift_for_data_augmentation = 5
train = ge_data.ge_train_dataset(ml_h5)
val = ge_data.ge_train_dataset(ml_h5)
batchsize = 128
train_iter = DataLoader(train, batchsize)
val_iter = DataLoader(val, batchsize, shuffle=False)

print('dataloader is available')