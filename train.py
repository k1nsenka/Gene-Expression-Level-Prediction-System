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


#ml_h5 = h5py.File('/Users/nemomac/Nextcloud/dataset/l131k_w128.h5')
#ml_h5 = h5py.File('/Users/nemomac/Nextcloud/dataset/l131k_w128.h5')
ml_h5 = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')

train_in = ml_h5['train_in']
train_out = ml_h5['train_out']

valid_in = ml_h5['valid_in']
valid_out = ml_h5['valid_out']

test_in = ml_h5['test_in']
test_out = ml_h5['test_out']

ratio = 1
train_ = train_in[:len(train_in)//ratio]
train_y = train_out[:len(train_out)//ratio]
valid_x = valid_in[:len(valid_in)//ratio]
valid_y = valid_out[:len(valid_out)//ratio]


max_shift_for_data_augmentation = 5
train = testdataset(train_in, train_out)
val = testdataset(valid_in, valid_out)


max_shift_for_data_augmentation = 5
train = testdataset(train_in, train_out)
val = testdataset(valid_in, valid_out)

batchsize = 8

train_iter = DataLoader(train, batchsize)
val_iter = DataLoader(val, batchsize, shuffle=False)
for i in train_iter:
    print(i)



lr = 0.001
n_epochs = 10
n_warmups = 0




train_model = Net()
train_model = train_model.to("cuda")
optimizer = optim.Adam(train_model.parameters(), lr, betas = (0.97, 0.98))

train_model.train()
for Epochs in range(n_epochs):
    for train_in, train_out in train_iter:
        out = train_model(train_in)
        loss = log_poisson_loss(out, train_out)
        acc = log_r2_score(out, train_out)
        train_model.zero_grad()
        loss.backward()
        optimizer.step()
