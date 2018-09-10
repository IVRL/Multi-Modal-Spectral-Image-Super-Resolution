import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import *
from dataset import Hdf5Dataset

# Training settings
parser = argparse.ArgumentParser(description="PIRM2018")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=600, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")

parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

parser.add_argument('--use-mask', default=False, type=bool, help='whether to use a mask when computing the error (default:False)')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = Hdf5Dataset()
    training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    
    valid_set = Hdf5Dataset(False)
    validation_data_loader = DataLoader(dataset=valid_set, num_workers=1, batch_size=1)

    print("===> Building model")
    model = ResidualLearningNet()
    mrae = MRAE()
    sid = SID()

    print("===> Setting GPU")
    model = model.cuda()
    mrae = mrae.cuda()
    sid = sid.cuda()
    
    # Loading previous models
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, mrae, sid, epoch)
        save_checkpoint(model, epoch)
        #validate(validation_data_loader, model, mrae, sid)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 50))
    return lr

def train(training_data_loader, optimizer, model, mrae, sid, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        data, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        data, target = torch.clamp(data / 65535.0, 0, 1), target / 65535.0
        output = torch.clamp(model(data), 0, 1)
        
        mask = None
        if opt.use_mask:
            mask = Variable(batch[2]).cuda()
        mrae_loss = mrae(output, target, mask) 
        sid_loss = sid(output, target, mask)
        loss = mrae_loss + sid_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f} + {:.10f} = {:.10f}".format(epoch, iteration, len(training_data_loader), mrae_loss.item(), sid_loss.item(), loss.item()))
            
def validate(validation_data_loader, model, mrae, sid):
    model.eval()
    total_mrae = 0
    total_sid = 0
    total_loss = 0
    print("===> Validation")
    for iteration, batch in enumerate(validation_data_loader, 1):
        data, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        data, target = torch.clamp(data / 65535.0, 0, 1), target / 65535.0
        output = torch.clamp(model(data), 0, 1)
        
        mask = None
        if opt.use_mask:
            mask = Variable(batch[2]).cuda()
        mrae_loss = mrae(output, target, mask) 
        sid_loss = sid(output, target, mask)
        loss = mrae_loss + sid_loss
        
        total_mrae += mrae_loss.item()
        total_sid += sid_loss.item()
        total_loss += loss.item()
        print("===> Image {}: Loss: {:.10f} + {:.10f} = {:.10f}".format(iteration,  mrae_loss.item(), sid_loss.item(), loss.item()))
        
    count = len(validation_data_loader)
    print("Average Loss: {:.10f} + {:.10f} = {:.10f}".format(total_mrae / count, total_sid / count, total_loss / count))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
