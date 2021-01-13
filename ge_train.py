import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
import h5py
import time
import sys
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

import ge_data
import ge_loss
import ge_nn


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        with open('train_log.txt', 'a') as f:
            f.write('save model\n')


def ge_train_fun_kfold(data, n_device, n_epochs, batchsize, n_targets, k_fold):
    print('calling dataloader ...')
    with open('train_log.txt', 'a') as f:
        f.write('calling dataloader ...\n')
    #モデル読み込み
    train_set = ge_data.ge_train_dataset(data)
    #train_iter = DataLoader(train_set, batchsize)
    device_str = 'cuda:{}'.format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print('used device : ', device)
    #損失関数
    loss_fun = nn.PoissonNLLLoss()
    #foldごとの学習の損失
    avg_kfold_train_loss = []
    #foldごとの検証の損失
    avg_kfold_valid_loss = []
    #交差検証用
    kf = KFold(n_splits = k_fold)
    for  _fold, (train_index, val_index) in enumerate(kf.split(train_set)):
        with open('train_log.txt', 'a') as f:
            f.write('fold{} start\n'.format(_fold))
        #交差検証用のtrain, validデータをロード
        train_loader = DataLoader(Subset(train_set,train_index), batch_size=batchsize, shuffle=True, num_workers=16)
        val_loader = DataLoader(Subset(train_set,val_index), batch_size=batchsize, shuffle=True, num_workers=16)
        #バッチごと学習の損失を追う
        train_losses = []
        #バッチごと検証の損失を追う
        valid_losses = []
        #epochごとの学習の損失
        avg_train_losses = []
        #epochごとの検証の損失
        avg_valid_losses = []
        #モデル定義
        ge_model = ge_nn.Net(n_targets=n_targets)
        ge_model.to(device)
        #最適化手法の選択
        optimizer = optim.Adam(ge_model.parameters())
        # initialize the early_stopping object
        early_stopping = EarlyStopping(verbose=True)
        for epoch in range(n_epochs):
            #学習
            ge_model.train()
            for train_in, train_out in tqdm(train_loader):
                #モデル入力
                train_in,  train_out = train_in.to(device), train_out.to(device)
                out = ge_model(train_in)
                #損失計算
                loss = loss_fun(out, train_out)
                ge_model.zero_grad()
                loss.backward()
                optimizer.step()
                #損失記録
                train_losses.append(loss.item())
            #検証
            ge_model.eval()
            for valid_in, valid_out in tqdm(val_loader):
                #モデル入力
                valid_in = valid_in.to(device)
                valid_out = valid_out.to(device)
                out = ge_model(valid_in)
                #損失計算
                loss = loss_fun(out, valid_out)
                #損失記録
                valid_losses.append(loss.item())
            #損失の平均をとる
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            #エポックごとの損失を保存していく
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            print('fold: {}/{} epoch: {}/{} train_loss: {:.6f} valid_loss: {:.6f}'.format(_fold+1, k_fold, epoch+1, n_epochs, train_loss, valid_loss))
            with open('train_log.txt', 'a') as f:
                f.write('fold: {}/{} epoch: {}/{} train_loss: {:.6f} valid_loss: {:.6f}\n'.format(_fold+1, k_fold, epoch+1, n_epochs, train_loss, valid_loss))
            #次のエポックのためにリセットする
            train_losses = []
            valid_losses = []
            #earlystopping
            early_stopping(valid_loss, ge_model, path='./model_checkpoint/checkpoint_fold{}.pth'.format(_fold))
            if early_stopping.early_stop:
                print('Early stopping')
                with open('train_log.txt', 'a') as f:
                    f.write('Early stopping\n')
                break
        #損失の平均をとる
        kfold_train_loss = np.average(avg_train_losses)
        kfold_valid_loss = np.average(avg_valid_losses)
        #foldごとの損失を保存していく
        avg_kfold_train_loss.append(kfold_train_loss)
        avg_kfold_valid_loss.append(kfold_valid_loss)
        #一番いいモデルをロードする。
        #ge_model.load_state_dict(torch.load('./model_checkpoint/checkpoint_fold{}.pth'.format(_fold)))
        #学習状況を可視化
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
        plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')
        #validlossの最低を検索する->earlystoppingに利用する
        minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5) # consistent scale
        plt.xlim(0, len(avg_train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig('./loss_plot/loss_plot_fold{}.png'.format(_fold), bbox_inches='tight')
        with open('train_log.txt', 'a') as f:
            f.write('fold{}, graph ploted\n'.format(_fold))
    print(avg_kfold_train_loss)
    print(avg_kfold_valid_loss)
    with open('./kfold_loss/kfold_train_loss.csv', 'w') as ft :
        writer = csv.writer(ft)
        writer.writerows(enumerate(avg_kfold_train_loss))
    with open('./kfold_loss/kfold_valid_loss.csv', 'w') as fv :
        writer = csv.writer(fv)
        writer.writerows(enumerate(avg_kfold_valid_loss))
    n_bestmodel_fold = avg_kfold_valid_loss.index(min(avg_kfold_valid_loss))
    return n_bestmodel_fold


def ge_train_fun_mse(data, n_device, n_epochs, batchsize, n_targets):
    print('calling dataloader ...')
    with open('train_log.txt', 'a') as f:
        f.write('calling dataloader ...\n')
    #モデル読み込み
    train_set = ge_data.ge_train_dataset(data)
    val_set = ge_data.ge_valid_dataset(data)
    device_str = 'cuda:{}'.format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print('used device : ', device)
    #損失関数
    loss_fun = nn.MSELoss()
    #train, validデータをロード
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=80)
    val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=True, num_workers=80)
    #バッチごと学習の損失を追う
    train_losses = []
    #バッチごと検証の損失を追う
    valid_losses = []
    #epochごとの学習の損失
    avg_train_losses = []
    #epochごとの検証の損失
    avg_valid_losses = []
    #モデル定義
    ge_model = ge_nn.Net(n_targets=n_targets)
    ge_model.to(device)
    #最適化手法の選択
    optimizer = optim.Adam(ge_model.parameters())
    # initialize the early_stopping object
    early_stopping = EarlyStopping(verbose=True)
    for epoch in range(n_epochs):
        #学習
        ge_model.train()
        for train_in, train_out in tqdm(train_loader):
            #モデル入力
            train_in,  train_out = train_in.to(device), train_out.to(device)
            out = ge_model(train_in)
            #損失計算
            loss = loss_fun(out, train_out)
            ge_model.zero_grad()
            loss.backward()
            optimizer.step()
            #損失記録
            train_losses.append(loss.item())
        #検証
        ge_model.eval()
        for valid_in, valid_out in tqdm(val_loader):
            #モデル入力
            valid_in = valid_in.to(device)
            valid_out = valid_out.to(device)
            out = ge_model(valid_in)
            #損失計算
            loss = loss_fun(out, valid_out)
            #損失記録
            valid_losses.append(loss.item())
        #損失の平均をとる
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        #エポックごとの損失を保存していく
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print('epoch: {}/{} train_loss: {:.6f} valid_loss: {:.6f}'.format(epoch+1, n_epochs, train_loss, valid_loss))
        with open('train_log.txt', 'a') as f:
            f.write('epoch: {}/{} train_loss: {:.6f} valid_loss: {:.6f}\n'.format(epoch+1, n_epochs, train_loss, valid_loss))
        #次のエポックのためにリセットする
        train_losses = []
        valid_losses = []
        #earlystopping
        early_stopping(valid_loss, ge_model, path='./mse/checkpoint_fold_mse.pth')
        if early_stopping.early_stop:
            print('Early stopping')
            with open('train_log.txt', 'a') as f:
                f.write('Early stopping\n')
            break
    #一番いいモデルをロードする。
    #ge_model.load_state_dict(torch.load('./model_checkpoint/checkpoint_fold{}.pth'.format(_fold)))
    #学習状況を可視化
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')
    #validlossの最低を検索する->earlystoppingに利用する
    minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('./mse/loss_plot_mse.png', bbox_inches='tight')
    with open('train_log.txt', 'a') as f:
        f.write('graph ploted\n')


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