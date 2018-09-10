import argparse, os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time, math, glob
import scipy.io as sio
from skimage.measure import compare_ssim, compare_psnr
from dataset import Hdf5Dataset
import h5py

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--model", default="checkpoint/model_epoch_600.pth", type=str, help="model path")
parser.add_argument("--results", default="validation", type=str, help="Result save location")

def get_output(data, model):
    data = Variable(data).cuda()
    data = torch.clamp(data, 0, 1)
    
    output = model(data)
    output = torch.clamp(output, 0, 1)
    output = output.detach().cpu().numpy()
    
    return output

opt = parser.parse_args()
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

model.cuda()
model.eval()

validate_images = glob.glob('../../data/track1/testing_lr/*.h5')

print("===> Validation")
for iteration, h5pyfilename in enumerate(validate_images, 1):
    basename = os.path.basename(h5pyfilename)[:-3]
    image = h5py.File(h5pyfilename, 'r')['data'][:] / 65535.0
    image_t = torch.from_numpy(image)
    
    output = get_output(image_t, model)
    
    output_g = np.copy(output)
    output_g[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output_g[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    
    print("===> Image %d" % iteration)

    np.savez('%s/%s.npz' % (opt.results, basename), out=output, out_g=output_g)
