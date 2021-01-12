import torch
import numpy as np
import h5py
import ignite.contrib.metrics.regression as regression
from sklearn.metrics import r2_score


#(バッチサイズ分, 1024, 4229)
batchsize = 64
size = 1024

data = h5py.File('/Users/nemomac/gelp/dataset/l131k_w128.h5', 'r')
train_out = data['train_out']
test_out = data['test_out']

x = train_out[:3]
y = train_out[1:4]
x = torch.Tensor(x)
y = torch.Tensor(y)
print(x.shape)
print(y.shape)
#print(regression.R2Score(x, y))
#z = r2_score(x, y, multioutput='variance_weighted')
#print(z)

'''
t = t.to("cpu")
log_x = log_x.to("cpu")
x = torch.exp(log_x)
x = log_x
'''

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
    score = r2_score(x_num, t_num)
    all_score += score
print(all_score / size)