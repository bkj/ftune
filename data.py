#!/usr/bin/env python

"""
    data.py
"""

import os
import h5py

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torchvision import transforms
from torchvision.datasets import ImageFolder

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.length = len(h5py.File(h5_path, 'r'))
    
    def __getitem__(self, index):
        h5_file = h5py.File(self.h5_path, 'r')
        
        record = h5_file[str(index)]
        res = (
            torch.from_numpy(record['data'].value),
            record['target'].value,
        )
        
        h5_file.close()
        return res
        
    def __len__(self):
        return self.length


dataset_stats = {
    "imagenet" : (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
}

def make_datasets(root, img_size=224, transforms=None):
    if transforms is None:
        transforms_train = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Are these reasonable?
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*dataset_stats['imagenet'])
        ])
        transforms_valid = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*dataset_stats['imagenet'])
        ])
    else:
        transforms_train, transforms_valid = transforms
    
    return {
        "train"       : ImageFolder(root=os.path.join(root, 'train'), transform=transforms_train),
        "train_fixed" : ImageFolder(root=os.path.join(root, 'train'), transform=transforms_valid),
        "val"         : ImageFolder(root=os.path.join(root, 'valid'), transform=transforms_valid),
    }


def make_dataloaders(datasets, train_batch_size=64, eval_batch_size=128, num_workers=8, seed=123, pin_memory=False):
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

