import sys
import os

# import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

# import numpy as np
import argparse
import json
# import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

writer = SummaryWriter()

def main():
    
    global args, best_prec1
    
    best_prec1 = 1e6
    best_prec2 = 1e6
    
    args = parser.parse_args()
#     args.original_lr = 1e-7
    args.original_lr = 1e-6  
    # 学习率改为1e-6
    
#     args.lr = 1e-7
    args.lr = 1e-6

    args.batch_size = 1
    args.momentum = 0.95

    args.decay = 5*1e-4
    args.start_epoch = 0
    # args.epochs = 400
    args.epochs = 200

    args.steps = [-1,1,100,150]
    args.scales = [1, 1, 0.1, 0.1]  # 学习率调整
    # args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
#     args.print_freq = 30
    args.print_freq = 100

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec2 = checkpoint['best_prec2']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    
    for epoch in range(args.start_epoch, args.epochs):

        start = time.time()
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch)
        prec1, mse = validate(val_list, model, criterion, epoch)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        best_prec2 = min(mse, best_prec2)
        writer.add_scalar('MAE(MSE)/mae', best_prec1, epoch)
        writer.add_scalar('MAE(MSE)/mse', mse, epoch)

        print(' * best MAE {mae:.3f},best MSE {mse:.3f} '
              .format(mae=best_prec1, mse=best_prec2))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec2': mse,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.task)
        
        during = time.time()-start
        print('Training complete in {:.0f}m {:.0f}s'.format(during/60, during % 60))


def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'MseLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    # writer.add_scalar('scalar/train_loss', loss, epoch)
    writer.add_scalar('Loss/train', losses.avg, epoch)


def validate(val_list, model, criterion, epoch):
    losses = AverageMeter()
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    mse = 0
    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())   # 平均绝对误差
        mse += (output.data.sum()-target.sum().type(torch.FloatTensor).cuda())**2   # 均方误差(每一项不是求绝对值,而是求平方)

    writer.add_scalar('Loss/val', losses.avg, epoch)
    mse = mse / len(test_loader)
    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f}, MSE {mse:.3f}'.format(mae=mae, mse=mse))

    return mae, mse
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        