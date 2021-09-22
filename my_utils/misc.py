'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import pdb
import torch

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'MovingAverage', 'AverageMeter_Mat', 'Timer']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
class MovingAverage(object):
    def __init__(self, length):
        self.length = length
        self.count = 0
        self.pointer = 0
        self.values = np.zeros(length)
        # self.avg = 0

    def update(self, val):
        self.values[self.pointer] = val
        self.pointer += 1
        if self.pointer == self.length:
            self.pointer = 0
        self.count += 1
        self.count = np.minimum(self.count, self.length)

    def avg(self):
        return self.values.sum() / float(self.count)

    def reset(self):
        self.count = 0
        self.pointer = 0
        # self.avg = 0
        self.values.fill(0)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
        # pdb.set_trace()
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_Mat(object):
    def __init__(self,number_ID):
        self.number_ID = number_ID
        self.reset()

    def reset(self):
        # self.sum = Variable(torch.Tensor(self.number_ID,64).fill_(0).cuda())
        # self.num = Variable(torch.Tensor(self.number_ID,64).fill_(0).cuda())
        self.center = Variable(torch.Tensor(self.number_ID,64).fill_(0).cuda(), requires_grad=False)
        # self.dif = Variable(torch.Tensor(self.number_ID,64).fill_(0).cuda())

        self.sum = torch.Tensor(self.number_ID,64).fill_(0).cuda()
        self.num = torch.Tensor(self.number_ID,64).fill_(0).cuda()
        # self.center = torch.Tensor(self.number_ID,64).fill_(0).cuda()
        # self.dif = torch.Tensor(self.number_ID,64).fill_(0).cuda()
        # self.sum = torch.Tensor(self.number_ID,64).fill_(0)
        # self.num = torch.Tensor(self.number_ID,64).fill_(0)
        # self.center = torch.Tensor(self.number_ID,64).fill_(0)
        # self.dif = torch.Tensor(self.number_ID,64).fill_(0)


    def update(self, SIR, ID, n):

        # pdb.set_trace()
        self.sum[ID,:] += SIR.data
        # pdb.set_trace()
        self.num[ID,:] += 1*n
        self.center[ID,:] = self.sum[ID] / self.num[ID]
        # self.dif[ID,:] = SIR - Variable(self.center[ID])
        # self.avg = 0.5*torch.mean(self.dif**2)

class Timer(object):
    def __init__(self):
        pass

    def reset(self):
        self.T = time.time()

    def time(self, reset=False):
        ti = time.time() - self.T
        if reset:
            self.reset()
        return ti