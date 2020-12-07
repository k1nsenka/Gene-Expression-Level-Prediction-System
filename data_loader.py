import h5py
import numpy as np


def data_loader(file_path):
    data_h5 = h5py.File('file_path', 'r')
    
    train_x = data_h5['train_in']
    train_y =  