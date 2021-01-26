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


def ge_test_extract_fun(data, n_device, batchsize, n_targets, model_path, n_extract):
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
            count = count + 1
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #グラフ描画
            out = torch.exp(out)
            if count == n_extract:
                test_out = test_out.to("cpu")
                out = out.to("cpu")
                with open('data_extract_log.txt', 'a') as f:
                    f.write('data{}:tensor detach numpy\n'.format(count))
                test_out = test_out.detach().numpy()
                out = out.detach().numpy()
                #testデータ番号に応じてcsvファイルにデータを抽出(4229, 1024)
                with open('data_extract_log.txt', 'a') as f:
                    f.write('data{}:test out data csv write\n'.format(count))
                with open('/home/abe/data/genome_data/data310/test_out/data_test_out{}.csv'.format(count), 'w') as fc :
                    writer = csv.writer(fc)
                    writer.writerows(test_out)
                with open('data_extract_log.txt', 'a') as f:
                    f.write('data{}:model out data csv write 2\n'.format(count))
                with open('/home/abe/data/genome_data/data310/data_out{}.csv'.format(count), 'w') as fc :
                    writer = csv.writer(fc)
                    writer.writerows(out)
            else :
                with open('data_extract_log.txt', 'a') as f:
                    f.write('data{}:data went through\n'.format(count))
    print('data extract finished')
    with open('data_extract_log.txt', 'a') as f:
        f.write('data extract finished'.format)


#データ
#ここを買えたらge_nnのn_targetsの数を変更してくださいseq->10, l131k_w128->4229
data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
#data = h5py.File('/home/abe/data/genome_data/seq.h5')
#data = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')
#data = h5py.File('/Users/nemomac/gelp/dataset/l131k_w128.h5')
n_targets = 4229

#使用GPU
#0-7
args = sys.argv
n_device = int(args[1])

#バッチサイズ
batchsize = 1

#抽出番号
n_extract = 310

#実行
model_path = './model_checkpoint/checkpoint_fold0.pth'
ge_test_extract_fun(data, n_device, batchsize, n_targets, model_path, n_extract)