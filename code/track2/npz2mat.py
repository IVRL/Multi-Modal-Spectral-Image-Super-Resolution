import glob
import numpy as np
from scipy.io import savemat
import argparse, os

parser = argparse.ArgumentParser(description='PIRM2018')
parser.add_argument('--dir', default='validation', type=str, help='Path to the npz files')
opt = parser.parse_args()

files = glob.glob('%s/*.npz' % opt.dir)
for file in files:
  a = np.load(file)
  savemat(file.replace('npz', 'mat'), a)
