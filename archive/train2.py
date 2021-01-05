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
ml_h5 = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
#ml_h5 = h5py.File('/home/abe/data/genome_data/seq.h5')
#ml_h5 = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')

train_in = ml_h5['train_in']
train_out = ml_h5['train_out']
valid_in = ml_h5['valid_in']
valid_out = ml_h5['valid_out']

print('calling dataloader ...')
max_shift_for_data_augmentation = 5
train = ge_data.ge_train_dataset1(train_in, train_out)
val = ge_data.ge_train_dataset1(valid_in, valid_out)
batchsize = 128
train_iter = DataLoader(train, batchsize)
val_iter = DataLoader(val, batchsize, shuffle=False)


####################################################################
lr = 0.001
n_epochs = 10
train_model = ge_nn.Net()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("used device : ", device)
train_model.to(device)
optimizer = optim.Adam(train_model.parameters(), lr, betas = (0.97, 0.98))
loss_fun = nn.PoissonNLLLoss()
loss_fun2 = nn.MSELoss()

train_model.train()
for epoch in range(n_epochs):
    counter = 0
    batch_loss = 0.0
    batch_acc = 0.0
    print('Epoch {}/{}'.format(epoch+1, n_epochs))
    print('------------------------------------------------')
    for train_in, train_out in train_iter:
        t1 = time.time()
        counter += counter
        #変数定義
        #モデル入力
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        out = train_model(train_in)
        #損失計算
        loss = loss_fun(out, train_out)
        mse_loss = loss_fun2(out, train_out)
        acc = ge_loss.log_r2_score(out, train_out)
        train_model.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        batch_acc += acc
        t2 = time.time()
        print('{} batch{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f} time {}'.format(epoch+1, counter,  loss, mse_loss, acc, t2-t1))
    print('------------------------------------------------')
    epoch_loss = batch_loss / batchsize
    epoch_acc = batch_acc / batchsize
    print('{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f}'.format(epoch+1, epoch_loss, mse_loss, epoch_acc))
    print('------------------------------------------------')
    print('------------------------------------------------')
    torch.save(train_model.state_dict(), "./params/model_epoch{}.pth".format(epoch))
