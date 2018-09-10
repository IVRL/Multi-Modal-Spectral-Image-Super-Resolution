import argparse, os
import torch
import h5py
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time, math, glob
import scipy.io as sio
from skimage.measure import compare_ssim, compare_psnr
from dataset import Hdf5Dataset


parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--firstmodel", default="../track1/checkpoint/model_epoch_600.pth", type=str, help="first model path")
parser.add_argument("--model", default="checkpoint/model_epoch_500.pth", type=str, help="model path")
parser.add_argument("--results", default="validation/", type=str, help="Result save location")


opt = parser.parse_args()
model1 = torch.load(opt.firstmodel, map_location=lambda storage, loc: storage)["model"]
model2 = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

validate_images = glob.glob('../../data/track2/testing_lr/*.h5')

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

model1.cuda()
model1.eval()
model2.cuda()
model2.eval()
summed_measures = None
print("===> Validation")
for iteration, h5pyfilename in enumerate(validate_images, 1):
    basename = os.path.basename(h5pyfilename)[:-3]

    image = h5py.File(h5pyfilename, 'r')['data'][:] / 65535.0
    image_t = torch.from_numpy(image)
    image_1 = h5py.File(h5pyfilename, 'r')['datatif'][:] / 65535.0
    image_tif = torch.from_numpy(image_1)
    
    data = Variable(image_t).cuda()
    data = torch.clamp(data, 0, 1)
    output1 = torch.clamp(model1(data), 0, 1) 
    output1 = output1.detach().cpu().numpy()
    output1[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output1[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    output1 = Variable(torch.from_numpy(output1)).cuda()

    datatif = Variable(image_tif).cuda()
    datatif = torch.clamp(datatif, 0, 1)
    output = torch.clamp(model2(output1, datatif), 0, 1)
    
    output = output.detach().cpu().numpy()
    output1 = output1.detach().cpu().numpy()

    output_g = np.copy(output)
    output_g[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output_g[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    
    print("===> Image %d" % iteration)

    np.savez('%s/%s.npz' % (opt.results, basename), out=output, out_g=output_g)
    