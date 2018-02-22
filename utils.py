#!/usr/bin/env python

"""
    utils.py
"""

import copy
import numpy as np

def fit_lr(model, dataloaders, mode='train', num_batches=np.inf, smooth_loss=False):
    assert mode in dataloaders, '%s not in loader' % mode
    
    model = copy.deepcopy(model)
    
    avg_mom  = 0.98
    avg_loss = 0.
    
    lr_hist, loss_hist = [], []
    for batch_idx, (data, target) in enumerate(dataloaders[mode]):
        
        model.set_progress(batch_idx)
        
        _, loss = model.train_batch(data, target)
        if smooth_loss:
            avg_loss    = avg_loss * avg_mom + loss * (1 - avg_mom)
            debias_loss = avg_loss / (1 - avg_mom ** (batch_idx + 1))
            loss_hist.append(debias_loss)
        else:
            loss_hist.append(loss)
        
        lr_hist.append(model.lr)
        
        if loss > np.min(loss_hist) * 4:
            break
    
    return np.vstack(lr_hist), loss_hist
