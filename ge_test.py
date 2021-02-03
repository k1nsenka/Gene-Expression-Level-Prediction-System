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
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=False, num_workers=50)
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
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=False, num_workers=50)
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


def ge_test_peason_fun(data, n_device, batchsize, n_targets, model_path):
    #データロード
    test_set = ge_data.ge_test_dataset(data)
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=False, num_workers=50)
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
    #テストデータ番号
    count = 0
    with torch.no_grad():
        for (test_in, test_out) in test_loader:
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #グラフ描画
            out = torch.exp(out)
            #平滑化、相関係数を計算
            test_out = test_out.to("cpu")
            out = out.to("cpu")
            test_out = torch.chunk(test_out, batchsize, dim=0)
            out = torch.chunk(out, batchsize, dim=0)
            #配列格納用
            smooth_out = []
            smooth_test_out = []
            test_score = []
            #バッチの中身一つずつについて計算していく
            for i in range(batchsize):
                count = count + 1
                if count == 761:
                    break
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:squeez\n'.format(count))
                t = torch.squeeze(test_out[i])
                o = torch.squeeze(out[i])
                #スムージング(1024, n_targets)
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:smoothing\n'.format(count))
                s_t = ge_loss.smoothing(t, n_targets)
                s_o = ge_loss.smoothing(o, n_targets)
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:detach\n'.format(count))
                #s_t = s_t.detach().numpy()
                #s_o = s_o.detach().numpy()
                #(1024, n_targets)testデータ番号に応じてcsvファイルにデータを格納、720ファイル*2できるはず
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:smoothing csv write\n'.format(count))
                #with open('/home/abe/data/genome_data/smoothing/test_out/smoothing_test_out{}.csv'.format(count), 'w') as fc :
                #    writer = csv.writer(fc)
                #    writer.writerows(s_t)
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:smoothing csv write 2\n'.format(count))
                #with open('/home/abe/data/genome_data/smoothing/out/smoothing_out{}.csv'.format(count), 'w') as fc :
                #    writer = csv.writer(fc)
                #    writer.writerows(s_o)
                #ピアソン相関の計算(n_targets)
                s_t = torch.tensor(s_t)
                s_o = torch.tensor(s_o)
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:pearson\n'.format(count))
                pearson = ge_loss.pearsonR(s_o, s_t, n_targets)
                test_score.append(pearson)
            #(batchsize, n_targets)ずつファイルに追記していく
            with open('pearson_test_log.txt', 'a') as f:
                f.write('data{}:pearson csv write\n'.format(count))
            print(len(test_score))
            with open('./smoothing/smoothing_pearsonr.csv', 'a') as fp :
                writer = csv.writer(fp)
                writer.writerows(test_score)
            avr_test_loss = np.average(test_loss)
            avr_test_score = np.mean(test_score)
    print('test data loss:{}, test r2 score:{}, \n max:{} index{}:, \n min:{} index{}'.format(avr_test_loss, avr_test_score, np.max(test_score), np.argmax(test_score), np.min(test_score), np.argmin(test_score)))
    with open('pearson_test_log.txt', 'a') as f:
        f.write('test data loss:{}, test r2 score:{}, max:{} index{}:, min:{} index{}'.format(avr_test_loss, avr_test_score, np.max(test_score), np.argmax(test_score), np.min(test_score), np.argmin(test_score)))



def ge_test_peason_raw_fun(data, n_device, batchsize, n_targets, model_path):
    #データロード
    test_set = ge_data.ge_test_dataset(data)
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=False, num_workers=50)
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
    #テストデータ番号
    count = 0
    with torch.no_grad():
        for (test_in, test_out) in test_loader:
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #グラフ描画
            out = torch.exp(out)
            #平滑化、相関係数を計算
            test_out = test_out.to("cpu")
            out = out.to("cpu")
            test_out = torch.chunk(test_out, batchsize, dim=0)
            out = torch.chunk(out, batchsize, dim=0)
            #配列格納用
            smooth_out = []
            smooth_test_out = []
            test_score = []
            #バッチの中身一つずつについて計算していく
            for i in range(batchsize):
                count = count + 1
                if count == 761:
                    break
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:squeez\n'.format(count))
                t = torch.squeeze(test_out[i])
                o = torch.squeeze(out[i])
                s_t = t
                s_o = o
                #ピアソン相関の計算(n_targets)
                s_t = torch.tensor(s_t)
                s_o = torch.tensor(s_o)
                with open('pearson_test_log.txt', 'a') as f:
                    f.write('data{}:pearson\n'.format(count))
                pearson = ge_loss.pearsonR(s_o, s_t, n_targets)
                test_score.append(pearson)
            #(batchsize, n_targets)ずつファイルに追記していく
            with open('pearson_test_log.txt', 'a') as f:
                f.write('data{}:pearson csv write\n'.format(count))
            print(len(test_score))
            with open('./smoothing/pearsonr.csv', 'a') as fp :
                writer = csv.writer(fp)
                writer.writerows(test_score)
            avr_test_loss = np.average(test_loss)
            avr_test_score = np.mean(test_score)
    print('test data loss:{}, test r2 score:{}, \n max:{} index{}:, \n min:{} index{}'.format(avr_test_loss, avr_test_score, np.max(test_score), np.argmax(test_score), np.min(test_score), np.argmin(test_score)))
    with open('pearson_test_log.txt', 'a') as f:
        f.write('test data loss:{}, test r2 score:{}, max:{} index{}:, min:{} index{}'.format(avr_test_loss, avr_test_score, np.max(test_score), np.argmax(test_score), np.min(test_score), np.argmin(test_score)))
