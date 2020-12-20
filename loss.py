import torch
import torch.functional as F
import math
import sklearn
import numpy as np
import cupy as cp

def log_poisson_loss(log_x, t):
    loss =  torch.mean(torch.exp(log_x) - t * log_x)
    t = t.astype(np.float32).to("cpu")
    #lossの最小値が0になるようにoffsetを引き算している。
    offset = torch.mean(cp.array(t - t * np.ma.log(t)))
    return loss - offset
