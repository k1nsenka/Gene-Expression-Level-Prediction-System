import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import csv
import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np


import ge_data
import ge_nn
import ge_loss


def ge_test_fun(data, n_device, batchsize, n_targets, model_path):
    #データロード
    test_set = ge_data.ge_test_dataset(data)
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=True, num_workers=50)
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    #損失関数
    loss_fun = nn.PoissonNLLLoss()
    #モデルの読み込み
    test_model = ge_nn.Net(n_targets=n_targets)
    test_model.to(device)
    test_model.load_state_dict(torch.load(model_path))
    test_model.eval()
    #損失の記録
    test_loss = []
    test_score = []
    count = 0
    with torch.no_grad():
        for (test_in, test_out) in test_loader:
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #score計算
            score = ge_loss.log_r2_score(out, test_out)
            test_score.append(score)
            #グラフ描画
            out = torch.exp(out)
            test_out = test_out.to("cpu")
            out = out.to("cpu")
        avr_test_loss = np.average(test_loss)
        avr_test_score = np.average(test_score)
    print('test data loss:{}, test r2 score:{}'.format(avr_test_loss, avr_test_score))
    with open('train_log.txt', 'a') as f:
        f.write('test data loss:{}, test r2 score:{}'.format(avr_test_loss, avr_test_score))


def ge_test_plot_fun(data, n_device, batchsize, n_targets, model_path):
    #データロード
    test_set = ge_data.ge_test_dataset(data)
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=True, num_workers=50)
    device_str = "cuda:{}".format(n_device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("used device : ", device)
    #損失関数
    loss_fun = nn.PoissonNLLLoss()
    #モデルの読み込み
    test_model = ge_nn.Net(n_targets=n_targets)
    test_model.to(device)
    test_model.load_state_dict(torch.load(model_path))
    test_model.eval()
    #損失の記録
    test_loss = []
    test_score = []
    count = 0
    with torch.no_grad():
        for (test_in, test_out) in test_loader:
            count = count + 1
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #score計算
            score = ge_loss.log_r2_score(out, test_out)
            test_score.append(score)
            #グラフ描画
            out = torch.exp(out)
            test_out = test_out.to("cpu")
            out = out.to("cpu")
            #データ抽出用
            cage = 3421
            dnase = 543
            H3K79me2 = 956
            H3K4me3 = 955
            H3K9ac = 1086
            if count == 310:#何番目のデータを見るか, test_data_label.bedを見て位置を決める
                #dnase
                plt.figure(figsize=(10,1))
                plt.bar(range(test_out.shape[1]), test_out[0, :, dnase])
                plt.savefig('result/test_out/test_out_dnase.png')
                plt.clf()
                plt.figure(figsize=(10,1))
                plt.bar(range(out.shape[1]), out[0, :, dnase])
                plt.savefig('result/model_out/model_out_dnase.png')
                plt.clf()
                #histone
                plt.figure(figsize=(10,1))
                plt.bar(range(test_out.shape[1]), test_out[0, :, H3K79me2])
                plt.savefig('result/test_out/test_out_H3K79me2.png')
                plt.clf()
                plt.figure(figsize=(10,1))
                plt.bar(range(out.shape[1]), out[0, :, H3K79me2])
                plt.savefig('result/model_out/model_out_H3K79me2.png')
                plt.clf()
                #histone
                plt.figure(figsize=(10,1))
                plt.bar(range(test_out.shape[1]), test_out[0, :, H3K4me3])
                plt.savefig('result/test_out/test_out_H3K4me3.png')
                plt.clf()
                plt.figure(figsize=(10,1))
                plt.bar(range(out.shape[1]), out[0, :, H3K4me3])
                plt.savefig('result/model_out/model_out_H3K4me3.png')
                plt.clf()
                #histone
                plt.figure(figsize=(10,1))
                plt.bar(range(test_out.shape[1]), test_out[0, :, H3K9ac])
                plt.savefig('result/test_out/test_out_H3K9ac.png')
                plt.clf()
                plt.figure(figsize=(10,1))
                plt.bar(range(out.shape[1]), out[0, :, H3K9ac])
                plt.savefig('result/model_out/model_out_H3K9ac.png')
                plt.clf()
                #CAGE
                plt.figure(figsize=(10,1))
                plt.bar(range(test_out.shape[1]), test_out[0, :, cage])
                plt.savefig('result/test_out/test_out_cage.png')
                plt.clf()
                plt.figure(figsize=(10,1))
                plt.bar(range(out.shape[1]), out[0, :, cage])
                plt.savefig('result/model_out/model_out_cage.png')
                plt.clf()
        avr_test_loss = np.average(test_loss)
        avr_test_score = np.average(test_score)
    print('test data loss:{}, test r2 score:{}'.format(avr_test_loss, avr_test_score))
