# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:17:53 2021

@author: Michel Frising
         michel.frising@uam.es
"""

def makeModel(config):
    if config['model_type'] == 'cvae':
        from .cvae import CVAE
        model = CVAE(config)
        
    if config['model_type'] == 'cinn':
        from .cinn import CINN
        cond_params = config['cond_params']
        dim_x = config['dim_x']
        dim_y = config['dim_y']
        if config['cond_net'] == 'dense':
            from .dense import Dense
            cond_params['layers'] = [dim_y] + cond_params['hidden_layers'] + [cond_params['out_dim']]
            cond_net = Dense(cond_params)
            
        elif config['cond_net'] == 'resNet':
            # ToDo: need to clean this up a bit
            from .resnet import ResNet
            cond_params['in_dim']=1
            cond_net = ResNet(**cond_params)
            cond_features = cond_params['out_dim']
        
        else:
            raise ValueError("Model not recognized")
        
        model = CINN()
        model.makeCondNet(cond_net, cond_features)
        model.makeINN(dim_x, **config['cinn_params'])
    
    return model, config