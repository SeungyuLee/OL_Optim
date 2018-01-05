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
		module.ol_update(int(bw_array[i]), threshold)
		i = i + 1

def update_and_test(bw_array, threshold=0.95):
	update(bw_array[0], threshold)
	return test()

num_layers = 38.0
threshold = 0.96

def evalAvgBw(array):
	avg_bw = (sum(array[0])/num_layers)*threshold + 16*(1-threshold)
	return avg_bw


def myf(array):
	acc = update_and_test(array)
	avg_bw = evalAvgBw(array)
	return (94.84 - acc)**2 - (2.5*(4.48 - avg_bw))**2
	
def main():
	import GPyOpt
	import GPy
	import numpy as np
# ground truth (full quantization) accuracy is 94.84%
	max_acc = 94.84

	space = [{'name': 'var_1', 'type': 'discrete', 'domain': (2,3,4,5), 'dimensionality': 38}]

	bo = GPyOpt.methods.BayesianOptimization(myf, space)

	"""
	constraints = [{'name': 'constr_1', 'constrain': '-update_and_test(x[:])+93.84'}]

	feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

	initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 20)

	print(initial_design)
	objective = GPyOpt.core.task.SingleObjective(evalAvgBw)
	model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=True)
	acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
	acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=acquisition_optimizer)
	evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

	bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)
	"""
	max_time = None
	max_iter = 30
	tolerance = 1e-8
	
	bo.run_optimization(max_iter = max_iter, max_time = max_time, eps=tolerance, verbosity=True)

	print(bo.x_opt)
	print(evalAvgBw([bo.x_opt]))
	print(update_and_test([bo.x_opt]))
	print(bo.fx_opt)

if __name__ == "__main__":
	main()
