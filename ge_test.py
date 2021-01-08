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
    test_loader = DataLoader(test_set, batch_size = batchsize, shuffle=True, num_workers=os.cpu_count())
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
    with torch.no_grad():
        for test_in, test_out in test_loader:
            #モデル入力
            test_in,  test_out = test_in.to(device), test_out.to(device)
            out = test_model(test_in)
            #損失計算
            loss = loss_fun(out, test_out)
            test_loss.append(loss.item())
            #score = 
        avr_test_loss = np.average(test_loss)
        #avr_test_score = 
    print('test data loss:{}, test r2 score:'.format(avr_test_loss))


