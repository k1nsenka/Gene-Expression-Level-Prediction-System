import h5py
import sys
import csv
import matplotlib.pyplot as plt

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

#バッチサイズ
batchsize = 64

#最適化手法
#0:Adam
#1:SGD
#2:RMSprop
#3:Adadelta
#4:AdamW
#5:Adagrad
#6:ASGD
#7:Adamax
n_optim = n_device

model_path = './model_checkpoint/checkpoint_fold0.pth'
ge_test.ge_test_fun(data, n_device, batchsize, n_targets, model_path)