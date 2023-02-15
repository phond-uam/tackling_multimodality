# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:18:53 2021

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
import torch.nn as nn

from collections import OrderedDict

from torch.nn.utils import weight_norm

from copy import deepcopy

ACTIVATION_MAPPING = OrderedDict(
    [
     ('sigmoid', nn.Sigmoid),
     ('tanh', nn.Tanh),
     ('relu', nn.ReLU),
     ('leaky_relu', nn.LeakyReLU),
     ]
    )

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu', bias=True, init='He', use_batchnorm=False, use_weightnorm=False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        if use_batchnorm:
            bias = False
        self.use_batchnorm = use_batchnorm
        self.linear = nn.Linear(input_dim, output_dim, bias=bias, **kwargs)
        if use_weightnorm:
            self.linear = weight_norm( self.linear )
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = None
        if activation is None:
            self.activation = None
        else:
            if activation in ACTIVATION_MAPPING:
                self.activation = ACTIVATION_MAPPING[activation]()
            else:
                raise ValueError(
                    "Unrecognized activation function: {}. Should contain one of {}".format(
                        activation, ", ".join(ACTIVATION_MAPPING.keys())
                    )
                )
        # init weights
        self.init_linear(init, activation)
        
    def init_linear(self, init, activation):
        # see here for more details: https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        if init=='He':
            if activation in ['relu', 'leaky_relu']:
                nn.init.kaiming_normal_( self.linear.weight, nonlinearity=activation)
            else:
                print(f"He initialization not supported for {activation}, falling back to normal")
                init = 'normal'
            
        if init=='uniform':
            nn.init.uniform_(self.linear.weight, -0.08, 0.08)
        elif init=='xavier':
            if activation is None:
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
            else:
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(activation))
        elif init=='normal':
            nn.init.normal_(self.linear.weight, 0, 0.05)
    
    def forward(self, x):
        x = self.activation(self.linear(x)) if self.activation is not None else self.linear(x)
        return self.bn(x) if self.use_batchnorm else x
        
class Dense(nn.Module):
    def __init__(self, model_config):
        super(Dense, self).__init__()
        layers = model_config.pop('layers')
        config = deepcopy(model_config)
        output_activation = config.pop('output_activation')
        config.pop('hidden_layers')
        modules = []
        for i in range(len(layers)-2):
            modules.append( LinearLayer(layers[i], layers[i+1], **config) )
        config['activation'] = output_activation
        # Weirdly enough this only works if I use batchnorm on the output layer
        config['use_batchnorm'] = True
        modules.append( LinearLayer(layers[-2], layers[-1], **config) )
        
        self.net = nn.Sequential( *modules )
    
    def forward(self, x):
        return self.net(x)