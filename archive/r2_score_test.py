import torch
import numpy as np
import h5py
import ignite.contrib.metrics.regression as regression
from sklearn.metrics import r2_score
import sys
import torch.nn as nn
import matplotlib.pyplot as plt

import ge_nn


#(バッチサイズ分, 1024, 4229)
batchsize = 64
size = 1024
n_targets = 4229

#デバイス設定
args = sys.argv
n_device = int(args[1])
device_str = "cuda:{}".format(n_device)
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print("used device : ", device)

#テストデータ
#data = h5py.File('/Users/nemomac/gelp/dataset/l131k_w128.h5', 'r')
data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
test_in = data['test_in']
test_out = data['test_out']


model_path = './model_checkpoint/checkpoint_fold0.pth'
test_model = ge_nn.Net(n_targets=n_targets)
test_model.load_state_dict(torch.load(model_path))
test_model.eval()

x = test_in[0:10]
y = test_out[0:10]
x = torch.Tensor(x)
y = torch.Tensor(y)
with torch.no_grad():
    x = test_model(x)
x = torch.exp(x)
print(x.shape)
print(y.shape)
#print(regression.R2Score(x, y))
#z = r2_score(x, y, multioutput='variance_weighted')
#print(z)
size = x.size(0)
#print(size)
t_temp = torch.chunk(x, size, dim=0)
x_temp = torch.chunk(y, size, dim=0)

all_score = 0.0
for i in range(size):
    t_num = torch.squeeze(t_temp[i])
    x_num = torch.squeeze(x_temp[i])
    #print(t_num.shape)
    #print(x_num.shape)
    #print(t_num)
    #print(x_num)
    t_num = t_num.detach().numpy()
    x_num = x_num.detach().numpy()
    #print(t_num.shape)
    #print(x_num.shape)
    #print(t_num)
    #print(x_num)
    score = r2_score(x_num, t_num, multioutput='variance_weighted')
    all_score += score
model_score = all_score / size
print(model_score)
#print(min(model_score))
#print(max(model_score))
loss_fun = nn.PoissonNLLLoss()
loss = loss_fun(x, y)
print(loss)



plt.bar(range(y.shape[1]), y[0, :, 0])
plt.savefig("data.png")
plt.clf()
plt.bar(range(x.shape[1]), x[0, :, 0])
plt.savefig("data_model_exp.png")
plt.clf()
