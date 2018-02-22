#!/usr/bin/env python

"""
    cub.py
"""

from __future__ import print_function, division

from rsub import *
from matplotlib import pyplot as plt

import os
import sys
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
basenet.helpers.set_seeds(123)

# --
# IO

# !! Copying from `fastai` for the time being
sys.path.append('/home/bjohnson/software/fastai')
from fastai.dataset import ImageClassifierData
from fastai.transforms import tfms_from_model, transforms_side_on

tfms  = tfms_from_model(resnet34, 224, aug_tfms=transforms_side_on, max_zoom=1.1)
data  = ImageClassifierData.from_paths('_data/cub_splits', tfms=tfms)

dataloaders = {
    "train_fixed" : data.fix_dl,
    "train"       : data.trn_dl,
    "val"         : data.val_dl
}

# >>
datasets = make_datasets(root='_data/cub_splits', img_size=224)
dataloaders = make_dataloaders(datasets)
# <<

# --
# Define model

pop_last_k  = 2
top_hidden  = 512
num_classes = 200

orig_model   = resnet34(pretrained=True)
orig_layers  = helpers.get_children(orig_model)
orig_layers  = orig_layers[:-pop_last_k]
num_features = helpers.get_num_features(orig_layers) * 2

top_layers = [
    AdaptiveMultiPool2d(output_size=(1, 1)),
    Flatten(),
    
    nn.BatchNorm1d(num_features=num_features),
    nn.Dropout(p=0.25),
    nn.Linear(in_features=num_features, out_features=top_hidden),
    nn.ReLU(),
    
    nn.BatchNorm1d(num_features=top_hidden),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=top_hidden, out_features=num_classes)
]

model = TopModel(
    conv=nn.Sequential(*orig_layers),
    classifier=nn.Sequential(*top_layers),
    loss_fn=F.cross_entropy,
).cuda().eval()

helpers.apply_init(model.classifier, torch.nn.init.kaiming_normal)

model.verbose = False

# --
# Precompute convolutional features

model.precompute_conv(dataloaders, mode='train_fixed', cache='./.precompute_conv')
model.precompute_conv(dataloaders, mode='val', cache='./.precompute_conv')

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

cacheloaders = model.get_precomputed_loaders()
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


# --
# Estimate optimal LR for end-to-end fine tuning

basenet.helpers.set_freeze(model.conv, False)
lr_mults = np.array([0.01, 0.1, 1.0])

conv_children = list(model.conv.children())
params = [
    {"params" : helpers.parameters_from_children(conv_children[:6])},
    {"params" : helpers.parameters_from_children(conv_children[6:])},
    {"params" : model.classifier.parameters()},
]

# >>
# !! This is recommended by fastai -- but it produces a wacky plot AFAICT

# lr_hist, loss_hist = LRFind.find(model, dataloaders, params=params, 
#     lr_init=lr_mults, mode='train', smooth_loss=True)

# opt_lr = LRFind.get_optimal_lr(lr_hist, loss_hist)

# _ = plt.plot(lr_hist[:,-1], loss_hist)
# _ = plt.xscale('log')
# show_plot()
# >>
opt_lr = opt_lr * 0.1 * lr_mults
# <<

# --
# Finetuning the whole network

lr_scheduler = LRSchedule.burnin_sgdr(lr_init=opt_lr, period_length=1, t_mult=2)
model.init_optimizer(
    opt=torch.optim.SGD,
    params=params,
    lr_scheduler=lr_scheduler,
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
