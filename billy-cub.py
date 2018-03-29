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
from basenet.lr import LRSchedule, LRFind

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

tfms  = tfms_from_model(resnet34, 448, aug_tfms=transforms_side_on, max_zoom=1.1)
data  = ImageClassifierData.from_paths('_data/cub_splits', tfms=tfms)

dataloaders = {
    "train_fixed" : data.fix_dl,
    "train"       : data.trn_dl,
    "val"         : data.val_dl
}

# >>
# datasets = make_datasets(root='_data/cub_splits', img_size=224)
# dataloaders = make_dataloaders(datasets)
# <<

# >>

class DelfLayer(basenet.BaseNet):
    def __init__(self, input_dim=512, num_kp_groups=10, kp_per_group=1, **kwargs):
        super(DelfLayer, self).__init__(**kwargs)
        
        self.num_kp_groups = num_kp_groups
        self.kp_per_group = kp_per_group
        
        self.conv = nn.Conv2d(input_dim, kp_per_group * num_kp_groups, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Map to num_kp_groups x kp_per_group
        x = self.conv(x)
        
        # For each spatial location, take max similarity w/ one of the class keypoints
        x = x.view((x.shape[0], self.num_kp_groups, self.kp_per_group, x.shape[-2], x.shape[-1]))
        x = x.max(dim=2)[0]
        
        # Threshold at 0
        x = F.relu(x)
        
        # Take average of positive entries across spatial dimensions
        # eg, each image is now a num_kp_groups x 1 vector, where each entry is the average maximum keypoint similarity
        x = x.mean(dim=-1).mean(dim=-1)
        
        # Normalize across classes
        x = F.normalize(x, dim=-1)
        
        return x

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

# Define architecture of classifier
top_layers = [
    DelfLayer(num_kp_groups=256, kp_per_group=16),
    # AdaptiveMultiPool2d(output_size=(1, 1)),
    # Flatten(),
    
    nn.BatchNorm1d(num_features=256),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=256, out_features=256),
    nn.ReLU(),
    
    nn.BatchNorm1d(num_features=256),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=256, out_features=num_classes),
    
    nn.LogSoftmax(), # !! **
]

# Stack classifier on top of resnet feature extractor
model = TopModel(
    conv=nn.Sequential(*orig_layers),
    classifier=nn.Sequential(*top_layers),
    loss_fn=F.nll_loss, # !! **
).cuda().eval()

# >>
# x = Variable(next(iter(dataloaders['train']))[0])
# model(x).shape
# <<

# Use non-default init for classifier weights
helpers.apply_init(model.classifier, torch.nn.init.kaiming_normal)

# --
# Precompute convolutional features

model.verbose = True
model.precompute_conv(dataloaders, mode='train_fixed', cache='./.precompute_conv.448')
model.precompute_conv(dataloaders, mode='val', cache='./.precompute_conv.448')

# --
# Estimate optimal LR for finetuning

model.use_conv = False
cacheloaders = model.get_precomputed_loaders()

lr_hist, loss_hist = LRFind.find(model, cacheloaders, mode='train_fixed', smooth_loss=True)
opt_lr = LRFind.get_optimal_lr(lr_hist, loss_hist)


# --
# Train w/ precomputed features + constant learning rate

num_epochs_classifier_precomputed = 3
num_epochs_classifier_augmented = 3
num_epochs_end2end = 15

model.init_optimizer(
    opt=torch.optim.SGD,
    params=[{"params" : list(model.classifier.parameters())}],
    lr_scheduler=LRSchedule.constant(lr_init=opt_lr),
    momentum=0.9
)

for epoch in range(num_epochs_classifier_precomputed):
    train = model.train_epoch(cacheloaders, mode='train_fixed')
    valid = model.eval_epoch(cacheloaders, mode='val')
    print({
        "stage"             : "classifier_precomputed",
        "epoch"             : epoch,
        "train_debias_loss" : train['debias_loss'],
        "valid_loss"        : np.mean(valid['loss']),
        "valid_acc"         : valid['acc']
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
    momentum=0.9
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

# >>
# !! This is recommended by fastai -- but it produces a wacky plot AFAICT

# lr_hist, loss_hist = LRFind.find(model, dataloaders, params=params, 
#     lr_init=lr_mults, mode='train', smooth_loss=True)

# opt_lr = LRFind.get_optimal_lr(lr_hist, loss_hist)

# >>
opt_lr = opt_lr * 0.1 
# <<

# --
# Finetuning the whole network

model.init_optimizer(
    opt=torch.optim.SGD,
    params=params,
    lr_scheduler=LRSchedule.burnin_sgdr(lr_init=opt_lr * lr_mults, period_length=1, t_mult=2),
    momentum=0.9
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
