import torch
import random
import numpy as np

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, max_shift):
        self.xs = xs
        self.ys = ys
        self.max_shift = max_shift

    def __len__(self):
        return len(self.xs)

    def get_example(self, i):
        # It applies following preprocesses:
        #     - Cropping
        #     - Random flip

        x = self.xs[i]
        y = self.ys[i]
        s = random.randint(-self.max_shift, self.max_shift)
        x = np.roll(x, s, axis=0)
        return x, y