import torch.utils.data as data
import torch
import h5py
from torchvision import transforms
from functools import partial
import numpy as np
import random
from scipy.ndimage.interpolation import rotate, zoom

class Hdf5Dataset(data.Dataset):
    def __init__(self, training=True):
        super(Hdf5Dataset, self).__init__()
        
        self.training = training
        
        val = ''
        if not training:
            val = '_v'
        
        base = 'data/'

        if not training:
            base = 'datafan/'

        self.hr_dataset = h5py.File(base + 'hr' + val + '.h5')['/data']
        self.out_dataset = h5py.File(base + 'out' + val + '.h5')['/data']
        self.replaced_dataset = h5py.File(base + 'replaced' + val + '.h5')['/data']
        self.hr_g_dataset = h5py.File(base + 'hr_g' + val + '.h5')['/data']

    def __getitem__(self, index):
        x, y = self.replaced_dataset[index], self.hr_dataset[index]
        return x, y
        
    def __len__(self):
        return self.hr_dataset.shape[0]
