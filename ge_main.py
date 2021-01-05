import h5py


import train


#データ
data = h5py.File('/home/abe/data/genome_data/l131k_w128.h5')
#ml_h5 = h5py.File('/home/abe/data/genome_data/seq.h5')
#ml_h5 = h5py.File('/Users/nemomac/gelp/dataset/seq.h5')

#使用GPU
n_device = 1

#学習率
lr = 0.001

#エポック数
n_epochs = 10

#バッチサイズ
batchsize = 128

#最適化手法のパラメータ
beta1 = 0.97
beta2 = 0.98

#モデル保存ディレクトリ
model_dir = 'params'

#学習実行
train_test.ge_train(data, n_device, lr, n_epochs, batchsize, beta1, beta2, model_dir)

#モデル評価