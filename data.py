#!/usr/bin/env python

"""
    data.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

# !! Untested
dataset_stats = {
    "imagenet" : (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
}

def make_datasets(root, img_size=224):
    transforms_valid = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*dataset_stats['imagenet'])
    ])
    
    transforms_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Are these reasonable?
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*dataset_stats['imagenet'])
    ])
    
    return {
        "train"       : ImageFolder(root=os.path.join(root, 'train'), transforms=transforms_train),
        "train_fixed" : ImageFolder(root=os.path.join(root, 'train'), transforms=transforms_valid),
        "valid"       : ImageFolder(root=os.path.join(root, 'valid'), transforms=transforms_valid),
    }


def make_dataloaders(datasets, train_batch_size, eval_batch_size, num_workers, seed, pin_memory):
    dataloaders = {}
    for dataset_name, dataset in datasets.items():
        if 'train' in dataset_name:
            batch_size = train_batch_size
        else:
            batch_size = eval_batch_size
        
        dataloaders[dataset_name] =  torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    
    return dataloaders

