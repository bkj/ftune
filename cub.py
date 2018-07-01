#!/usr/bin/env python

"""
    cub.py
"""

from __future__ import print_function, division

from rsub import *
from matplotlib import pyplot as plt

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torchvision
from torchvision.models.resnet import resnet34

import basenet
from basenet import helpers
from basenet.hp_schedule import HPSchedule, HPFind

from models import TopModel
from data import make_datasets, make_dataloaders
from layers import AdaptiveMultiPool2d, Flatten

torch.backends.cudnn.deterministic = True

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

args = parse_args()

# --
# IO

# !! Copying from `fastai` for the time being
sys.path.append('/home/bjohnson/software/fastai')
from fastai.dataset import ImageClassifierData
from fastai.transforms import tfms_from_model, transforms_side_on

basenet.helpers.set_seeds(args.seed)

tfms  = tfms_from_model(resnet34, 224, aug_tfms=transforms_side_on, max_zoom=1.1)
data  = ImageClassifierData.from_paths('_data/cub_splits', tfms=tfms)

dataloaders = {
    "train_fixed" : data.fix_dl,
    "train"       : data.trn_dl,
    "val"         : data.val_dl
}

# # <<

# # Dump train
# all_data, all_target = [], []
# for i, (data, target) in enumerate(dataloaders['train_fixed']):
#     print(i)
#     data = helpers.to_numpy(data)
#     target = helpers.to_numpy(target)
#     all_data.append(data)
#     all_target.append(target)

# all_data = np.concatenate(all_data)
# all_target = np.hstack(all_target)

# sel = np.random.permutation(all_data.shape[0])
# all_data, all_target = all_data[sel], all_target[sel]
# np.save('_data/cub_train_X', all_data)
# np.save('_data/cub_train_y', all_target)

# # Dump val
# all_data, all_target = [], []
# for i, (data, target) in enumerate(dataloaders['val']):
#     print(i)
#     data = helpers.to_numpy(data)
#     target = helpers.to_numpy(target)
#     all_data.append(data)
#     all_target.append(target)

# all_data = np.concatenate(all_data)
# all_target = np.hstack(all_target)

# sel = np.random.permutation(all_data.shape[0])
# all_data, all_target = all_data[sel], all_target[sel]
# np.save('_data/cub_val_X', all_data)
# np.save('_data/cub_val_y', all_target)

# # >>

# >>
# datasets = make_datasets(root='_data/cub_splits', img_size=224)
# dataloaders = make_dataloaders(datasets)
# <<

# --
# Define model

pop_last_k  = 2   # Drop last 2 layers of resnet
top_hidden  = 512 # Number of channels in resnet output
num_classes = 200 # Number of classes in CUB dataset

orig_model   = resnet34(pretrained=True)
orig_layers  = helpers.get_children(orig_model)
orig_layers  = orig_layers[:-pop_last_k]
num_features = helpers.get_num_features(orig_layers) * 2

# >>

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


# <<

# Define architecture of classifier
top_layers = [
    # AdaptiveMultiPool2d(output_size=(1, 1)),
    # Flatten(),
    
    # nn.BatchNorm1d(num_features=num_features),
    # nn.Dropout(p=0.25),
    # nn.Linear(in_features=num_features, out_features=top_hidden),
    # nn.ReLU(),
    
    # nn.BatchNorm1d(num_features=top_hidden),
    # nn.Dropout(p=0.5),
    # nn.Linear(in_features=top_hidden, out_features=num_classes),
    
    # nn.LogSoftmax(), # !! **
            
    # PreActBlock(num_features // 2, num_features // 2),
    AdaptiveMultiPool2d(output_size=(1, 1)),
    Flatten(),
    nn.BatchNorm1d(num_features),
    nn.Linear(in_features=num_features, out_features=num_classes),
    nn.LogSoftmax(),
]

# Stack classifier on top of resnet feature extractor
model = TopModel(
    conv=nn.Sequential(*orig_layers),
    classifier=nn.Sequential(*top_layers),
    loss_fn=F.nll_loss, # !! **
).to('cuda').eval()

# Use non-default init for classifier weights
for child in model.classifier.children():
    try:
        helpers.apply_init(child, torch.nn.init.kaiming_normal)
    except:
        print('error')

# --
# Precompute convolutional features

model.verbose = True
model.use_classifier = False
model.precompute_features(dataloaders, mode='train_fixed', cache='_results/precomputed/cub/conv')
model.precompute_features(dataloaders, mode='val', cache='_results/precomputed/cub/conv')
model.use_classifier = True

# --
# Estimate optimal LR for finetuning

model.use_conv = False
cacheloaders = model.get_precomputed_loaders(cache='_results/precomputed/conv')

# lr_hist, loss_hist = LRFind.find(model, cacheloaders, mode='train_fixed', smooth_loss=True)
# opt_lr = LRFind.get_optimal_lr(lr_hist, loss_hist)
opt_lr = 0.2

# --
# Train w/ precomputed features + constant learning rate

num_epochs_classifier_precomputed = 50
num_epochs_classifier_augmented = 3
num_epochs_end2end = 15

model.verbose = False

model.init_optimizer(
    opt=torch.optim.SGD,
    params=[{"params" : list(model.classifier.parameters())}],
    lr_scheduler=LRSchedule.sgdr(lr_init=0.1, period_length=10, t_mult=1),
    momentum=0.9,
)

for epoch in range(num_epochs_classifier_precomputed):
    train = model.train_epoch(cacheloaders, mode='train_fixed')
    valid = model.eval_epoch(cacheloaders, mode='val')
    val_test = model.eval_epoch(cacheloaders, mode='val_test')
    test_test = model.eval_epoch(cacheloaders, mode='test_test')
    print({
        "stage"             : "classifier_precomputed",
        "epoch"             : epoch,
        "train_debias_loss" : train['debias_loss'],
        "valid_loss"        : np.mean(valid['loss']),
        "valid_acc"         : valid['acc'],
        
        "val_test_loss"   : np.mean(val_test['loss']),
        "val_test_acc"    : val_test['acc'],
        "test_test_loss"  : np.mean(test_test['loss']),
        "test_test_acc"   : test_test['acc'],
    })
    sys.stdout.flush()


# --
# Training w/ data augmentation + SGDR

model.use_conv = True
basenet.helpers.set_freeze(model.conv, True)

model.init_optimizer(
    opt=torch.optim.SGD,
    params=[{"params" : list(model.classifier.parameters())}],
    lr_scheduler=LRSchedule.burnin_sgdr(lr_init=opt_lr, period_length=1),
    momentum=0.9,
)

for epoch in range(num_epochs_classifier_augmented):
    train = model.train_epoch(dataloaders, mode='train')
    valid = model.eval_epoch(dataloaders, mode='val')
    print({
        "stage"             : "classifier_augmented",
        "epoch"             : epoch,
        "train_debias_loss" : train['debias_loss'],
        "valid_loss"        : np.mean(valid['loss']),
        "valid_acc"         : valid['acc']
    })
    sys.stdout.flush()


# --
# Estimate optimal LR for end-to-end fine tuning

basenet.helpers.set_freeze(model.conv, False)

conv_children = list(model.conv.children())
params = [
    {"params" : helpers.parameters_from_children(conv_children[:6])},
    {"params" : helpers.parameters_from_children(conv_children[6:])},
    {"params" : model.classifier.parameters()},
]

lr_mults = np.array([0.01, 0.1, 1.0])
opt_lr *= 0.1

# --
# Finetuning the whole network

model.init_optimizer(
    opt=torch.optim.SGD,
    params=params,
    lr_scheduler=LRSchedule.burnin_sgdr(lr_init=opt_lr * lr_mults, period_length=1, t_mult=2),
    momentum=0.9,
)

for epoch in range(num_epochs_end2end):
    train = model.train_epoch(dataloaders, mode='train')
    valid = model.eval_epoch(dataloaders, mode='val')
    print({
        "stage"             : "end2end",
        "epoch"             : epoch,
        "train_debias_loss" : train['debias_loss'],
        "valid_loss"        : np.mean(valid['loss']),
        "valid_acc"         : valid['acc']
    })
    sys.stdout.flush()


# 224x224 images
# ~ 0.738

# 448x448 images
# {'stage': 'end2end', 'epoch': 14, 'train_debias_loss': 0.14390290099964395, 'valid_loss': 0.7823975482484796, 'valid_acc': 0.7982395581636176}
