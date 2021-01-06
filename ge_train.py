import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import h5py
import time
import sys
from tqdm import tqdm
import pandas as pd

import ge_data
import ge_loss
import ge_nn

def ge_train_fun(data, n_device, n_epochs, batchsize, model_dir):
    print('calling dataloader ...')
    train_set = ge_data.ge_train_dataset(data)
    train_iter = DataLoader(train_set, batchsize)
    ####################################################################
    train_model = ge_nn.Net()
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    train_model.to(device)
    optimizer = optim.Adam(train_model.parameters())
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
            #acc = ge_loss.log_r2_score(out, train_out)
            acc = 0.0
            train_model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            batch_acc += acc
            t2 = time.time()
            #print('{} batch{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f} time {}'.format(epoch+1, counter,  loss, mse_loss, acc, t2-t1))
        print('------------------------------------------------')
        epoch_loss = batch_loss / batchsize
        epoch_acc = batch_acc / batchsize
        print('{} {:.4f} {:.4f} {:.4f}'.format(epoch+1, epoch_loss, mse_loss, epoch_acc))
        print('------------------------------------------------')
        print('------------------------------------------------')
        torch.save(train_model.state_dict(), "./" + model_dir + "/model_epoch{}.pth".format(epoch))


def ge_train_fun_optim(data, n_device, n_epochs, batchsize, n_optim, model_dir):
    #print('calling dataloader ...')
    #モデル読み込み
    train_set = ge_data.ge_train_dataset(data)
    #test = ge_data.hogehoge
    train_iter = DataLoader(train_set, batchsize)
    #モデル定義
    train_model = ge_nn.Net()
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    #print("used device : ", device)
    train_model.to(device)
    #最適化手法
    if n_optim == 0:
        optimizer = optim.Adam(train_model.parameters())
    elif n_optim == 1:
        optimizer = optim.SGD(train_model.parameters(), 0.01)
    elif n_optim == 2:
        optimizer = optim.RMSprop(train_model.parameters())
    elif n_optim == 3:
        optimizer = optim.Adadelta(train_model.parameters())
    elif n_optim == 4:
        optimizer = optim.AdamW(train_model.parameters())
    elif n_optim == 5:
        optimizer = optim.Adagrad(train_model.parameters())
    elif n_optim == 6:
        optimizer = optim.ASGD(train_model.parameters())
    elif n_optim == 7:
        optimizer = optim.Adamax(train_model.parameters())
    else :
        print('please input optimizer')
        sys.exit(1)
    loss_fun = nn.PoissonNLLLoss()
    loss_fun2 = nn.MSELoss()

    train_model.train()
    print('epoch poissonLoss mseLoss Acc')
    for epoch in range(n_epochs):
        counter = 0
        batch_loss = 0.0
        batch_acc = 0.0
        #print('Epoch {}/{}'.format(epoch+1, n_epochs))
        #print('------------------------------------------------')
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
            #acc = ge_loss.log_r2_score(out, train_out)
            acc = 0.0
            train_model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss
            batch_acc += acc
            t2 = time.time()
            #print('{} batch{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f} time {}'.format(epoch+1, counter,  loss, mse_loss, acc, t2-t1))
        #print('------------------------------------------------')
        epoch_loss = batch_loss / batchsize
        epoch_acc = batch_acc / batchsize
        print('{} {:.4f} {:.4f} {:.4f}'.format(epoch+1, epoch_loss, mse_loss, epoch_acc))
        #print('------------------------------------------------')
        #print('------------------------------------------------')
        torch.save(train_model.state_dict(), "./" + model_dir + "/model_optim{}_epoch{}.pth".format(n_optim, epoch))


def ge_train_fun_kfold(data, n_device, n_epochs, batchsize, n_optim, model_dir, k_fold):
    print('calling dataloader ...')
    #モデル読み込み
    train_set = ge_data.ge_train_dataset(data)
    train_iter = DataLoader(train_set, batchsize)
    #モデル定義
    train_model = ge_nn.Net()
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    train_model.to(device)
    #最適化手法
    if n_optim == 0:
        optimizer = optim.Adam(train_model.parameters())
    elif n_optim == 1:
        optimizer = optim.SGD(train_model.parameters(), 0.01)
    elif n_optim == 2:
        optimizer = optim.RMSprop(train_model.parameters())
    elif n_optim == 3:
        optimizer = optim.Adadelta(train_model.parameters())
    elif n_optim == 4:
        optimizer = optim.AdamW(train_model.parameters())
    elif n_optim == 5:
        optimizer = optim.Adagrad(train_model.parameters())
    elif n_optim == 6:
        optimizer = optim.ASGD(train_model.parameters())
    elif n_optim == 7:
        optimizer = optim.Adamax(train_model.parameters())
    else :
        print('please input optimizer')
        sys.exit(1)
    #損失関数
    loss_fun = nn.PoissonNLLLoss()
    loss_fun2 = nn.MSELoss()
    #交差検証用
    train_score = pd.Series()
    val_score = pd.Series()
    
    total_size = len(train_set)
    fraction = 1/k_fold
    seg = int(total_size * fraction)

    train_model.train()
    #print('epoch poissonLoss mseLoss Acc')
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
        #  print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
        #       % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = Subset(train_set,train_indices)
        val_set = Subset(train_set,val_indices)
        
        #print(len(train_set),len(val_set))
        #print()
        
        train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=50, shuffle=True, num_workers=4)
        
        train_acc = train(res_model,criterion,optimizer,train_loader,epoch=1)
        train_score.at[i] = train_acc
        val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.at[i] = val_acc
    
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
            #acc = ge_loss.log_r2_score(out, train_out)
            acc = 0.0
            train_model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss
            batch_acc += acc
            t2 = time.time()
            print('{} batch{} poissonLoss: {:.4f} mseLoss: {:.4f} Acc: {:.4f} time {}'.format(epoch+1, counter,  loss, mse_loss, acc, t2-t1))
        print('------------------------------------------------')
        epoch_loss = batch_loss / batchsize
        epoch_acc = batch_acc / batchsize
        print('{} {:.4f} {:.4f} {:.4f}'.format(epoch+1, epoch_loss, mse_loss, epoch_acc))
        print('------------------------------------------------')
        print('------------------------------------------------')
        #torch.save(train_model.state_dict(), "./" + model_dir + "/model_optim{}_epoch{}.pth".format(n_optim, epoch))


def crossvalid(model=None, criterion=None, optimizer=None, dataset=None,k_fold=5):
    train_score = pd.Series()
    val_score = pd.Series()
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
#         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
#               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        
#         print(len(train_set),len(val_set))
#         print()
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=True, num_workers=4)
        train_acc = train(res_model,criterion,optimizer,train_loader,epoch=1)
        train_score.at[i] = train_acc
        val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.at[i] = val_acc
    
    return train_score,val_score
        

train_score,val_score = crossvalid(res_model, criterion, optimizer, dataset=tiny_dataset)