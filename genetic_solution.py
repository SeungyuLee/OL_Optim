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

# ground truth (full quantization) accuracy is 94.84%
max_acc = 94.84
num_layers = 38.0
threshold = 0.95

import random
from deap import base, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 2, 4)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, int(num_layers))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalBWMin(individual):
	avg_bw = (sum(individual)/num_layers)*threshold + 16*(1-threshold)
	return avg_bw,

def feasible(individual):
	update(individual, threshold)
	test_acc = test()
	if test_acc > max_acc-1: return True
	return False

penalty = 8

toolbox.register("evaluate", evalBWMin)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=2, up=8, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, penalty))

def main():
	random.seed(64)

	pop_num = 20
	gen_num = 50
	CXPB, MUTPB = 0.5, 0.2

	pop = toolbox.population(n=pop_num)

	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		print (fit)
		ind.fitness.values = fit

	fits = [ind.fitness.values[0] for ind in pop]
	
	g = 0

	while g < gen_num:
		g = g + 1
		print("-- Generation %i --" % g)
		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))

		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			print(fit)
			ind.fitness.values = fit

		pop[:] = offspring

		fits = [ind.fitness.values[0] for ind in pop]

		length = len(pop)
		mean = sum(fits) / length

		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)

	print(pop)

	best_ind = tools.selBest(pop, 5)
	
	for i in range(5):
		update(best_ind[i], threshold)
		test_acc = test()
		print("Best individual is %s, %s, %s" % (best_ind[i], best_ind[i].fitness.values, test_acc))


main()
