import h5py
import sys
import csv
import matplotlib.pyplot as plt
import time

import ge_train
import ge_test


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

#学習率
#lr = 0.001

#エポック数
n_epochs = 200

#バッチサイズ
batchsize = 64

#交差検証分割数
k_fold = 6

t1 = time.time()

bestmodel_number = ge_train.ge_train_fun_kfold(data, n_device, n_epochs, batchsize, n_targets, k_fold)
print(bestmodel_number)

model_path = './model_checkpoint/checkpoint_fold{}.pth'.format(bestmodel_number)

#モデル評価
ge_test.ge_test_fun(data, n_device, batchsize, n_targets, model_path)

t2 = time.time()
print('time :{}'.format(t2-t1))