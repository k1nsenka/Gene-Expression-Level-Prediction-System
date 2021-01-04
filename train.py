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

print('importing data ...')
train_in = ml_h5['train_in']
train_out = ml_h5['train_out']
print('training data done ...')
valid_in = ml_h5['valid_in']
valid_out = ml_h5['valid_out']
print('valid data done ...')

#test_in = ml_h5['test_in']
#test_out = ml_h5['test_out']

print('transform data ...')
ratio = 1
train_in = train_in[:len(train_in)//ratio]
train_in = train_in.astype(np.float32)
train_out = train_out[:len(train_out)//ratio]
train_out = train_out.astype(np.float32)
print('training data done ...')
valid_in = valid_in[:len(valid_in)//ratio]
valid_in = valid_in.astype(np.float32)
valid_out = valid_out[:len(valid_out)//ratio]
valid_out = valid_out.astype(np.float32)
print('valid data done ...')

print('calling dataloader ...')
max_shift_for_data_augmentation = 5
train = ge_data.testdataset(train_in, train_out)
val = ge_data.testdataset(valid_in, valid_out)
batchsize = 512
train_iter = DataLoader(train, batchsize)
val_iter = DataLoader(val, batchsize, shuffle=False)


####################################################################
lr = 0.001
n_epochs = 10
train_model = ge_nn.Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("used device : ", device)
train_model.to(device)
optimizer = optim.Adam(train_model.parameters(), lr, betas = (0.97, 0.98))
loss_fun = nn.PoissonNLLLoss()
loss_fun2 = nn.MSELoss()

train_model.train()
for epoch in range(n_epochs):
    print('Epoch {}/{}'.format(epoch+1, n_epochs))
    print('------------------------------------------------')
    for train_in, train_out in train_iter:
        #変数定義
        batch_loss = 0.0
        batch_acc = 0.0
        #モデル入力
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        out = train_model(train_in)
        #損失計算
        loss = loss_fun(out, train_out)
        mse_loss = loss_fun2(out, train_out)
        #acc = ge_loss.log_r2_score(torch.log(out), train_out)
        train_model.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss
        print('{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f}'.format(epoch+1, batch_loss, mse_loss, batch_acc))
    print('------------------------------------------------')
    print('{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f}'.format(epoch+1, batch_loss, mse_loss, batch_acc))
    print('------------------------------------------------')
    print('------------------------------------------------')
    torch.save(train_model.state_dict(), "./params/model.pth")
