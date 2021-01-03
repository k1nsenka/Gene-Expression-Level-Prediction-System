import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py


import ge_dataset
import ge_loss
import ge_nn


ml_h5 = h5py.File('/Users/nemomac/Nextcloud/dataset/l131k_w128.h5')
#ml_h5 = h5py.File('/Users/nemomac/Nextcloud/dataset/l131k_w128.h5')
#ml_h5 = h5py.File('/Users/nemomac/Nextcloud/dataset/l131k_w128.h5')

train_x = ml_h5['train_in']
train_y = ml_h5['train_out']

valid_x = ml_h5['valid_in']
valid_y = ml_h5['valid_out']

test_x = ml_h5['test_in']
test_y = ml_h5['test_out']

ratio = 1
train_x = train_x[:len(train_x)//ratio]
train_y = train_y[:len(train_y)//ratio]
valid_x = valid_x[:len(valid_x)//ratio]
valid_y = valid_y[:len(valid_y)//ratio]


max_shift_for_data_augmentation = 5
train = dataset.PreprocessedDataset(train_x, train_y, max_shift_for_data_augmentation)

batchsize = 8

train_iter = DataLoader(train, batchsize)
val_iter = DataLoader(val, batchsize, repeat=False, shuffle=False)

lr = 0.001
n_epochs = 10
n_warmups = 0




#model####################################################################
train_model = ge_nn.Net()
train_model = train_model.to("cuda")
optimizer = optim.Adam(train_model.parameters(), lr, betas = (0.97, 0.98))
####################################################################

train_model.train()


for Epochs in range(n_epochs):
    for Batch in enumerate(train_iter):
        y = train_model(train_x)
        loss = loss.log_poisson_loss(y, train_y)
        train_model.zero_grad()
        loss.backward()
        optimizer.step()