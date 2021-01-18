import h5py
import sys
import csv

import ge_train


#データ
#ここを買えたらge_nnのn_targetsの数を変更してくださいseq->10, l131k_w128->4229
#data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
data = h5py.File('/home/abe/data/genome_data/seq.h5')
#data = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')
#data = h5py.File('/Users/nemomac/gelp/dataset/l131k_w128.h5')

#使用GPU
#0-7
args = sys.argv
n_device = int(args[1])

#学習率
#lr = 0.001

#エポック数
n_epochs = 2

#バッチサイズ
batchsize = 128

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

#モデル保存ディレクトリ
model_dir = 'poissonloss'

with open('./train_result_optim/result{}.csv'.format(n_device), 'w') as f :
    #標準出力先をファイルに変更
    sys.stdout = f
    #学習実行
    ge_train.ge_train_fun_optim(data, n_device, n_epochs, batchsize, n_optim, model_dir)

#モデル評価