import torch
import torch.functional as F
import torch.nn as nn
import math
import sklearn
import numpy as np
#import cupy as cp

from sklearn.metrics import r2_score


def log_poisson_loss(log_x, t):
    loss =  torch.mean(torch.exp(log_x) - t * log_x)
    #lossの最小値が0になるようにoffsetを引き算している。
    offset = torch.mean(t - t * torch.log(t))
    return loss - offset


def log_r2_score(log_x, t):
    t = t.to("cpu")
    log_x = log_x.to("cpu")
    x = torch.exp(log_x)
    x = log_x

    size = t.size(0)
    t_temp = torch.chunk(t, size, dim=0)
    x_temp = torch.chunk(x, size, dim=0)

    all_score = 0.0
    for i in range(size):
        t_num = torch.squeeze(t_temp[i])
        x_num = torch.squeeze(x_temp[i])
        t_num = t_num.detach().numpy()
        x_num = x_num.detach().numpy()
        score = r2_score(x_num, t_num, multioutput='variance_weighted')
        all_score += score
    return all_score / size

def pearsonR(output, target, n_targets):
    cost = []
    for i in range(n_targets):
        x = output[i]
        y = target[i]
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost_temp = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        #cost = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))
        cost_temp= cost_temp.detach().numpy()
        cost.append(cost_temp)
    return cost


def smoothing(sample_raw, n_targets):
    #移動平均の範囲
    window = 20
    #w:(1024)
    w = np.ones(window)/window
    #(1024, n_targets)
    avr_data = []
    for i in range(n_targets):
        sample = sample_raw[:, i]
        sample_avr = np.convolve(sample, w, mode='same')
        avr_data.append(sample_avr)
    return avr_data