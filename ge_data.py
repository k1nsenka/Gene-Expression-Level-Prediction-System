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