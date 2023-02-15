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

ACTIVATION_MAPPING = OrderedDict(
    [
     ('sigmoid', nn.Sigmoid),
     ('tanh', nn.Tanh),
     ('relu', nn.ReLU),
     ('leaky_relu', nn.LeakyReLU),
     ]
    )

class LinearLayer(nn.Module):
    """
    If alpha and beta are None, respectively, it means that they will be trainable.
    """
    def __init__(self, input_dim, output_dim, activation='relu', bias=True, init='He', use_batchnorm=False, device=None, dtype=None):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias, device=device, dtype=dtype)
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)
        if activation!='linear':
            if activation in ACTIVATION_MAPPING:
                self.activation = ACTIVATION_MAPPING[activation]()
            else:
                raise ValueError(
                    "Unrecognized activation function: {}. Should contain one of {}".format(
                        activation, ", ".join(ACTIVATION_MAPPING.keys())
                    )
                )
        else:
            self.activation = None
        # ToDo: dig deeper into parameter initialization
        # self.reset_parameters()
        if init=='He' and activation in ['relu', 'leaky_relu']:
            nn.init.kaiming_normal_( self.linear.weight, nonlinearity=activation)
        else:
            print(f"He initialization not supported for {activation}, falling back to normal")
            init = 'normal'
            
        if init=='uniform':
            self.linear.weight.data = 0.08*(1-2*torch.rand_like(self.linear.weight))
        elif init=='xavier':
            if self.activation is None:
                nn.init.xavier_uniform_(self.linear.weight, gain = nn.init.calculate_gain('linear'))
            else:
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(activation))
        elif init=='normal':
            nn.init.normal_(self.linear.weight, 0, 0.05)
                    
    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.linear(x))
        else:
            return self.linear(x)

def build_layer(dim_in, dim_out, activation, use_weightnorm, use_batchnorm, bias):
    layer = []
    if use_weightnorm:
        layer.append( weight_norm(nn.Linear(dim_in, dim_out, bias=bias)) )
    else:
        layer.append( nn.Linear(dim_in, dim_out, bias=bias) )
    if use_batchnorm:
        layer.append( nn.BatchNorm1d(dim_out) )
    if activation != 'linear':
        if activation in ACTIVATION_MAPPING:
            layer.append( ACTIVATION_MAPPING[activation]() )
        else:
            raise ValueError(
                "Unrecognized activation function: {}. Should contain one of {}".format(
                    activation, ", ".join(ACTIVATION_MAPPING.keys())
                )
            )
    return layer

def init_linear(linear, init, activation):
    # see here for more details:
    if init=='He' and activation in ['relu', 'leaky_relu']:
        nn.init.kaiming_normal_( linear.weight, nonlinearity=activation)
    else:
        print(f"He initialization not supported for {activation}, falling back to normal")
        init = 'normal'
        
    if init=='uniform':
        nn.init.xavier_uniform_(linear.weight, -0.08, 0.08)
    elif init=='xavier':
        if activation is None:
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('linear'))
        else:
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain(activation))
    elif init=='normal':
        nn.init.normal_(linear.weight, 0, 0.05)
            
        
class Dense(nn.Module):
    def __init__(self, model_config):
        super(Dense, self).__init__()
        layers = model_config.pop('layers')
        output_activation = model_config.pop('output_activation')
        modules = []
        for i in range(len(layers)-2):
            modules += build_layer(layers[i], layers[i+1], **model_config)
        
        model_config['activation'] = output_activation
        modules += build_layer(layers[-2], layers[-1], **model_config)
        
        for m in modules:
            if isinstance(module, nn.Linear):
        
        self.net = nn.Sequential(*net)
        # for m in modules:
#         if isinstance(module, nn.Linear):
    
    def forward(self, x):
        return self.net(x)