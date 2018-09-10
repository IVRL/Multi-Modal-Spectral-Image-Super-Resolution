import argparse, os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time, math, glob
import scipy.io as sio
from skimage.measure import compare_ssim, compare_psnr
from dataset import Hdf5Dataset


parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--model", default="checkpoint/model_epoch_600.pth", type=str, help="model path")
parser.add_argument("--results", default="results", type=str, help="Result save location")

def MSE(gt, rc):
    return np.mean((gt - rc) ** 2)

def PSNR(gt, rc):
    mse = MSE(gt, rc)
    pmax = 65536
    return 20 * np.log10(pmax / np.sqrt(mse + 1e-3))

def MRAE(gt, rc):
    return np.mean(np.abs(gt - rc) / (gt + 1e-3))

def SID(gt, rc):
    N = gt.shape[0]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(rc[i] * np.log10((rc[i] + 1e-3) / (gt[i] + 1e-3))) +
                        np.sum(gt[i] * np.log10((gt[i] + 1e-3) / (rc[i] + 1e-3))))
    return err.mean()

def APPSA(gt, rc):
    nom = np.sum(gt * rc, axis=0)
    denom = np.linalg.norm(gt, axis=0) * np.linalg.norm(rc, axis=0)
    
    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)
    
    return np.sum(appsa) / (gt.shape[1] * gt.shape[0])

def SSIM(gt, rc):
    return compare_ssim(gt, rc)

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
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

valid_set = Hdf5Dataset(False)
validation_data_loader = DataLoader(dataset=valid_set, num_workers=1, batch_size=1)

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

model.cuda()
model.eval()
summed_measures = None
print("===> Validation")
for iteration, batch in enumerate(validation_data_loader, 1):
    data = Variable(batch[0]).cuda()
    data = torch.clamp(data / 65535.0, 0, 1)
    output = torch.clamp(model(data), 0, 1) 
    
    output = output.detach().cpu().numpy()
    output = output[0]
    gt = batch[1][0].numpy() / 65535.0
    
    
    hr_measures = {k:np.array(func(gt, output)) for (k, func) in measures.items()}

    print("===> Image %d" % iteration)
    print(hr_measures)

    if summed_measures is None:
        summed_measures = hr_measures
    else:
        summed_measures = {k:v+hr_measures[k] for (k, v) in summed_measures.items()}


summed_measures = {k:v/len(validation_data_loader) for (k, v) in summed_measures.items()}
print('Average Measures')
print(summed_measures)
