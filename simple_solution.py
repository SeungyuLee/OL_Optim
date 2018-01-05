'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/SSD/cifar10_torch', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory = True)

testset = torchvision.datasets.CIFAR10(root='/SSD/cifar10_torch', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory = True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_350_lr0.1.t7')
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
           % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	
    # Save checkpoint.
    acc = 100.*correct/total
    return (acc)

def update(bw_array, threshold):
	i = 0 	
	
	for name, module in net.named_ol_layers():
		module.ol_update(bw_array[i], threshold)
		i = i + 1


def main():
# ground truth (full quantization) accuracy is 94.84%
	max_acc = 94.84
	num_layers = 38.0

# simple solution stage 1
	bw = 8
	test_acc = 0
	while(1):
		bw_array = [bw] * int(num_layers)
		update(bw_array, 0.99)
		test_acc = test()
		if(test_acc < max_acc - 1): break
		bw = bw - 1

	bw_stage1 = bw
	
# simple solution stage 2-1
	solution1 = {'bitwidth': None, 'threshold': None, 'accuracy': None, 'avg_bw' : None}

	threshold = 0.98
	while(1):
		update(bw_array, threshold)
		test_acc = test()
		if(test_acc > max_acc -1): break
		threshold = threshold - 0.01

	solution1['bitwidth'] = bw_stage1
	solution1['threshold'] = threshold
	solution1['accuracy'] = test_acc
	solution1['avg_bw'] = bw_stage1*threshold + 16*(1-threshold)
	
# simple solution stage 2-2
	threshold = 0.99
	solution2 = {'bitwidth': None, 'threshold': None, 'accuracy': None, 'avg_bw' : None}
	
	i = 0
	while(1):
		bw_array[i] = 8
		update(bw_array, threshold)
		test_acc = test()
		if(test_acc > max_acc-1): break
		i = i + 1

	solution2['avg_bw'] = (8*threshold + 16*(1-threshold)) * ((i+1)/num_layers) + (bw*threshold + 16*(1-threshold)) * ((37-i)/num_layers)
	solution2['bitwidth'] = bw_array
	solution2['threshold'] = threshold
	solution2['accuracy'] = test_acc

	print (solution1)
	print (solution2)


main()
