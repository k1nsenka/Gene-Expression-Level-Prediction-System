import torch
import random
import numpy as np
import h5py
from torch.utils import data



class PreprocessedDataset(data.Dataset):
    def __init__(self, xs, ys, max_shift):
        self.xs = xs
        self.ys = ys
        self.max_shift = max_shift

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        # It applies following preprocesses:
        #     - Cropping
        #     - Random flip

        x = self.xs[i]
        y = self.ys[i]
        s = random.randint(-self.max_shift, self.max_shift)
        x = np.roll(x, s, axis=0)
        return x, y

class testdataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = self.xs[i]
        y = self.ys[i]
        return x, y


class ge_train_dataset(torch.utils.data.Dataset):
    def __init__(self, ml_h5):
        self.ml_h5 = ml_h5
        print('importing train data ...')
        self.train_in = ml_h5['train_in']
        self.train_out = ml_h5['train_out']
        print('done ...')

    def __len__(self):
        return len(self.train_in)

    def __getitem__(self, i):
        x = self.train_in[i]
        x = x.astype(np.float32)
        y = self.train_out[i]
        y = y.astype(np.float32)
        return x, y

class ge_valid_dataset(torch.utils.data.Dataset):
    def __init__(self, ml_h5):
        self.ml_h5 = ml_h5
        print('importing valid data ...')
        self.valid_in = ml_h5['valid_in']
        self.valid_out = ml_h5['valid_out']
        print('done ...')

    def __len__(self):
        return len(self.train_in)

    def __getitem__(self, i):
        x = self.valid_in[i]
        x = x.astype(np.float32)
        y = self.valid_out[i]
        y = y.astype(np.float32)
        return x, y

