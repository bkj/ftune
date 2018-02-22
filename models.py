#!/usr/bin/env python

"""
    models.py
"""

from __future__ import print_function, division

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import basenet

class TopModel(basenet.BaseNet):
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
    
    def precompute_conv(self, dataloaders, mode, cache=None):
        if (cache is not None) and os.path.exists(os.path.join(cache, mode)):
                print('precompute_conv: loading %s' % os.path.join(cache, mode), file=sys.stderr)
                self._precomputed[mode] = {
                    "data"    : torch.load(os.path.join(cache, mode, 'data')),
                    "targets" : torch.load(os.path.join(cache, mode, 'targets')),
                }
        else:
            _ = self.eval()
            tmp = {"data" : [], "targets" : []}
            
            gen = dataloaders[mode]
            if self.verbose:
                gen = tqdm(gen, total=len(gen))
            
            self.use_conv = True
            self.use_classifier = False
            
            for data, target in gen:
                data = Variable(data, volatile=True).cuda()
                output = self(data)
                tmp['data'].append(basenet.helpers.to_numpy(output))
                tmp['targets'].append(basenet.helpers.to_numpy(target))
            
            tmp['data']    = torch.Tensor(np.vstack(tmp['data']))
            tmp['targets'] = torch.LongTensor(np.hstack(tmp['targets']))
            
            self._precomputed[mode] = tmp
            
            self.use_conv = True
            self.use_classifier = True
            
            if cache is not None:
                print('precompute_conv: saving %s' % os.path.join(cache, mode), file=sys.stderr)
                if not os.path.exists(os.path.join(cache, mode)):
                    os.makedirs(os.path.join(cache, mode))
                
                torch.save(tmp['data'], os.path.join(cache, mode, 'data'))
                torch.save(tmp['targets'], os.path.join(cache, mode, 'targets'))
    
    def get_precomputed_loaders(self, batch_size=64):
        out = {}
        for mode, tmp in self._precomputed.items():
            out[mode] = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tmp['data'], tmp['targets']),
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle='train' in mode
            )
        
        return out