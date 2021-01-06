import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py

import time
import timeit

import ge_data
import ge_loss
import ge_nn

print('opening file ...')
#ml_h5 = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
ml_h5 = h5py.File('/home/abe/data/genome_data/seq.h5')
#ml_h5 = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')

print('calling dataloader ...')
max_shift_for_data_augmentation = 5
train = ge_data.ge_train_dataset(ml_h5)
batchsize = 64
train_iter = DataLoader(train, batchsize)


####################################################################
lr = 0.001
n_epochs = 10
train_model = ge_nn.Net()
device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("used device : ", device)
train_model.to(device)
optimizer = optim.Adam(train_model.parameters(), lr, betas = (0.97, 0.98))
loss_fun = nn.PoissonNLLLoss()
loss_fun2 = nn.MSELoss()

train_model.train()
for epoch in range(n_epochs):
    for train_in, train_out in train_iter:
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        out = train_model(train_in)
        #loss = loss_fun(out, train_out)
        #mse_loss = loss_fun2(out, train_out)
        #acc = ge_loss.log_r2_score(out, train_out)
        train_model.zero_grad()
        #loss.backward()
        optimizer.step()