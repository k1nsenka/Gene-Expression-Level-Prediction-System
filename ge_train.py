import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import h5py
import time
from tqdm import tqdm

import ge_data
import ge_loss
import ge_nn


def ge_train_fun(data, n_device, lr, n_epochs, batchsize, beta1, beta2, model_dir):
    print('calling dataloader ...')
    train = ge_data.ge_train_dataset(data)
    val = ge_data.ge_train_dataset(data)
    #test = ge_data.hogehoge
    train_iter = DataLoader(train, batchsize)
    val_iter = DataLoader(val, batchsize, shuffle=False)
    ####################################################################
    train_model = ge_nn.Net()
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    train_model.to(device)
    optimizer = optim.Adam(train_model.parameters(), lr, betas = (beta1, beta2))
    loss_fun = nn.PoissonNLLLoss()
    loss_fun2 = nn.MSELoss()

    train_model.train()
    for epoch in range(n_epochs):
        counter = 0
        batch_loss = 0.0
        batch_acc = 0.0
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        print('------------------------------------------------')
        for train_in, train_out in tqdm(train_iter):
            t1 = time.time()
            counter = counter + 1
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
            batch_loss += loss
            batch_acc += acc
            t2 = time.time()
            #print('{} batch{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f} time {}'.format(epoch+1, counter,  loss, mse_loss, acc, t2-t1))
        print('------------------------------------------------')
        epoch_loss = batch_loss / batchsize
        epoch_acc = batch_acc / batchsize
        print('{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f}'.format(epoch+1, epoch_loss, mse_loss, epoch_acc))
        print('------------------------------------------------')
        print('------------------------------------------------')
        torch.save(train_model.state_dict(), "./" + model_dir + "/model_epoch{}.pth".format(epoch))