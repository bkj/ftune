#!/usr/bin/env python

"""
    models.py
"""

from __future__ import print_function, division

import os
import sys
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import basenet
from basenet import helpers

from data import H5Dataset

class PrecomputeMixin(object):
    def precompute_features(self, dataloaders, mode, cache, force=False):
        if not os.path.exists(cache):
            os.makedirs(cache)
            
        cache_path = os.path.join(cache, mode + '.h5')
        if os.path.exists(cache_path) and (not force):
            print('precompute_conv: using cache %s' % cache_path, file=sys.stderr)
        else:
            
            _ = self.eval()
            
            cache_file = h5py.File(cache_path, 'w', libver='latest')
            data, targets = self.predict(dataloaders, mode=mode)
            data, targets = helpers.to_numpy(data), helpers.to_numpy(targets)
            for i,(d,t) in enumerate(zip(data, targets)):
                cache_file['%d/data' % i] = d
                cache_file['%d/target' % i] = t
            
            cache_file.flush()
            cache_file.close()
    
    def get_precomputed_loaders(self, cache, batch_size=64, num_workers=8, **kwargs):
        loaders = {}
        for cache_path in glob(os.path.join(cache, '*.h5')):
            print("get_precomputed_loaders: loading cache %s" % cache_path, file=sys.stderr)
            loaders[os.path.basename(cache_path).split('.')[0]] = torch.utils.data.DataLoader(
                H5Dataset(cache_path),
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs
            )
        
        return loaders

class StackModel(basenet.BaseNet, PrecomputeMixin):
    def __init__(self, groups, **kwargs):
        super(StackModel, self).__init__(**kwargs)
        
        self.groups = nn.ModuleList(groups)
        self.mask = [1] * len(groups)
    
    def forward(self, x):
        for m, group in zip(self.mask, self.groups):
            if m:
                x = group(x)
        
        return x

class TopModel(basenet.BaseNet, PrecomputeMixin):
    def __init__(self, conv, classifier, **kwargs):
        super(TopModel, self).__init__(**kwargs)
        
        self.conv = conv
        self.classifier = classifier
        
        self.use_conv       = True
        self.use_classifier = True
        self._precomputed = {}
    
    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        
        if self.use_classifier:
            x = self.classifier(x)
        
        return x