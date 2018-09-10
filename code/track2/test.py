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
import time

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--firstmodel", default="../track1/checkpoint/model_epoch_600.pth", type=str, help="first model path")
parser.add_argument("--model", default="checkpoint/model_epoch_500.pth", type=str, help="model path")
parser.add_argument("--results", default="results", type=str, help="Result save location")

def MSE(gt, rc):
    return np.mean((gt - rc) ** 2)

def PSNR(gt, rc):
    mse = MSE(gt, rc)
    pmax = 65536
    return 20 * np.log10(pmax / np.sqrt(mse + 1e-3))

def MRAE(gt, rc):
    return np.mean(np.abs(gt - rc) / (gt + 1.0))

def SID(gt, rc):
    N = gt.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(rc[:,:,i] * np.log10((rc[:,:,i] + 1e-3)/(gt[:,:,i] + 1e-3))) +
                     np.sum(gt[:,:,i] * np.log10((gt[:,:,i] + 1e-3)/(rc[:,:,i] + 1e-3))))
    SIDs = err / (gt.shape[1] * gt.shape[0])
    return np.mean(SIDs)

def APPSA(gt, rc):
    nom = np.sum(gt * rc, axis=0)
    denom = np.linalg.norm(gt, axis=0) * np.linalg.norm(rc, axis=0)
    
    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)
    
    return np.sum(appsa) / (gt.shape[1] * gt.shape[0])

def SSIM(gt, rc):
    return compare_ssim(gt, rc)


def get_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn=nn*s;
        pp += nn
    return pp

# First element is the ground truth, second is the prediction
measures = {
    'APPSA': APPSA,
    'SID' : SID,
    'PSNR': PSNR,
    'SSIM': compare_ssim,
    'MSE' : MSE,
    'MRAE': MRAE
}

opt = parser.parse_args()
model1 = torch.load(opt.firstmodel, map_location=lambda storage, loc: storage)["model"]
model2 = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

validate_images = glob.glob('/scratch/pirm/data/track2/validation_hd5/*.h5')

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

model1.cuda()
model1.eval()
print(get_parameters(model1))
model2.cuda()
model2.eval()
print(get_parameters(model2))
summed_measures = None
print("===> Validation")
for iteration, h5pyfilename in enumerate(validate_images, 1):
    start = time.time()
    basename = os.path.basename(h5pyfilename)[:-3]

    image = h5py.File(h5pyfilename, 'r')['data'][:] / 65535.0
    image_t = torch.from_numpy(image)
    image_1 = h5py.File(h5pyfilename, 'r')['datatif'][:] / 65535.0
    image_tif = torch.from_numpy(image_1)
    hrimg = h5py.File(h5pyfilename, 'r')['hrdata'][:]
    image_hr = torch.from_numpy(hrimg)

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

    output[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    output1 = output1[0] * 65535.0
    output = output[0] * 65535.0
    gt = image_hr[0].numpy() #/ 65535.0

    #print(output)
    
    hr_measures = {k:np.array(func(gt, output)) for (k, func) in measures.items()}
    origin_measures = {k:np.array(func(gt, output1)) for (k, func) in measures.items()}

    print("===> Image %d" % iteration)
    print(origin_measures)
    print(hr_measures)
    end = time.time()
    print(end - start)

    if summed_measures is None:
        origin_summed = origin_measures
        summed_measures = hr_measures
    else:
        summed_measures = {k:v+hr_measures[k] for (k, v) in summed_measures.items()}
        origin_summed = {k:v+origin_measures[k] for (k, v) in origin_summed.items()}


summed_measures = {k:v/10 for (k, v) in summed_measures.items()}
origin_summed = {k:v/10 for (k, v) in origin_summed.items()}
print('Average Measures')
print(origin_summed)
print(summed_measures)
