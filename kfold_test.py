import h5py
import sys
import csv
import matplotlib.pyplot as plt

import ge_train


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

#交差検証分割数
k_fold = 6

nice_model, train_loss, valid_loss, kfole_loss = ge_train.ge_train_fun_kfold2(data, n_device, n_epochs, batchsize, n_targets, model_dir, k_fold)


#学習状況を可視化
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')


#validlossの裁定を検索する
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')


#モデル評価