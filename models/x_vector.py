#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch.nn as nn
from models.tdnn import TDNN
import torch
import torch.nn.functional as F

def masked_stats(tensor, mask):
    mean = torch.div(torch.sum(tensor*mask, dim=1),torch.sum(mask, dim=1))
    var = torch.square(tensor-mean)
    var = torch.sum(var*mask, dim=1)
    var = torch.div(var, torch.sum(mask, dim=1)+1.0e-8)
    std = torch.sqrt(var)

    return mean, std

class X_vector(nn.Module):
    '''
    context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
    context size 3 and dilation 2 is equivalent to [-2, 0, 2]
    context size 1 and dilation 1 is equivalent to [0]
    '''

    def __init__(self, input_dim = 60):
        super(X_vector, self).__init__()
        #self.mtl=mtl

        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=4, dilation=4,dropout_p=0.5)
        #self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment5 = nn.Linear(512, 1500)
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)
        #self.output = nn.Linear(512, num_classes)
        #if mtl > 0: # mtl = # classes for another objective
        #    self.output_mtl = nn.Linear(512, mtl)

    def forward(self, inputs, mask=None):
        tdnn1_out = self.tdnn1(inputs)
        return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        out = self.segment5(tdnn4_out) # (b, t, f)
        ### Stat Pool
        if mask is None:
            mean = torch.mean(out,1) # (b, f)
            std = torch.std(out,1) # (b, f)
        else:
            mean, std = masked_stats(out, mask)
        stat_pooling = torch.cat((mean,std),1) # (b, fx2)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        #predictions = self.output(x_vec)
        #if self.mtl > 0:
        #    predictions_mtl = self.output_mtl(x_vec)
        #    return predictions, predictions_mtl, x_vec
        #else:
        #    return predictions,x_vec
        return x_vec, stat_pooling
