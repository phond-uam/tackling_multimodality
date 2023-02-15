# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 18:15:08 2022

@author: Michel Frising
         michel.frising@uam.es
The cVAE class consisting of an encoder, decoder and the normalizing flow
"""
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from models.dense import Dense
from torch.nn.utils import weight_norm

log2pi = torch.FloatTensor([np.log(2*np.pi)])

class CVAE(nn.Module):
    def __init__(self, config):
        # dim_x, dim_y, dim_z, hidden_size=512, hidden_layers=10
        super(CVAE, self).__init__()
        dim_x = config['dim_x']
        dim_y = config['dim_y']
        dim_z = config['dim_z']
        config['encoder_config']['layers'] = [dim_x+dim_y]+config['encoder_config']['hidden_layers']+[2*dim_z]
        config['decoder_config']['layers'] = [dim_z+dim_y]+config['decoder_config']['hidden_layers']+[2*dim_x]
        self.encoder  = Dense(config['encoder_config']) # will be split into sigma and scale
        self.decoder  = Dense(config['decoder_config'])
        self.alpha = config['alpha']
        self.dim_z = config['dim_z']
        
    def _encode(self, x, y):
        # run through net
        xy = torch.cat( [x, y], dim=-1)
        out = self.encoder(xy)
        # construct z_0
        mu, logvar = torch.chunk(out, chunks=2, dim=-1)
        sigma = torch.exp(0.5*logvar) # using logvar has the additional advantage that sigma will
        # always be possible. Otherwise we might want to use a softplus function or something similar
        eps = torch.normal(mean=torch.zeros_like(sigma), std=1)
        z_0 = mu + eps*sigma
        kl_divergence = (-0.5*(1 + logvar - mu.pow(2) - logvar.exp()))
        
        return z_0, kl_divergence.sum(-1)

    def _decode(self, z, y, x):
        # pass zk_y through the decoder
        zy = torch.cat( [z, y], dim=-1)
        out = self.decoder(zy)
        # use the reparametrization trick on output, this seems more stable
        mu, logvar = torch.chunk(out, chunks=2, dim=-1)
        sigma = torch.exp(0.5*logvar)
        eps = torch.normal(mean=torch.zeros_like(mu), std=1)
        x_gen = mu + eps*sigma
        log_pxz = -dist.Normal(mu, sigma).log_prob(x)
        # If we choose the output distribution to be Gaussian, we can actually
        # calculate log_pzkx as the MSE error. However, the variance in that 
        # case is 1
        return x_gen, log_pxz.sum(-1)
    
    def forward(self, x, y):
        z_0, kl_divergence = self._encode(x, y)
        # decode
        x, log_pxz = self._decode(z_0, y, x)
        return dict(x=x,
                    loss=(self.alpha*kl_divergence+log_pxz).mean(),
                    kl_div=kl_divergence.mean(),
                    log_pxz=log_pxz.mean())
    
    @torch.no_grad()
    def sample(self, x, y):
        # construct z from encoder. Since the dataset has multimodality, the 
        # assumption of a Gaussian does not approximate well the latent space
        # constructing z from the encoder yields better results
        xy = torch.cat( [x, y], dim=-1)
        out = self.encoder(xy)
        mu, logvar = torch.chunk(out, chunks=2, dim=-1)
        sigma = torch.exp(0.5*logvar)
        eps = torch.normal(mean=torch.zeros_like(sigma), std=1).to(sigma.device)
        z = mu.expand(sigma.shape[0],-1)+eps*sigma
        zy = torch.cat( [z, y.expand(sigma.shape[0],-1)], dim=-1)
        # pass zk_y through the decoder 
        out = self.decoder(zy)
        # use the reparametrization trick again or MSE
        mu, logvar = torch.chunk(out, chunks=2, dim=-1)
        sigma = torch.exp(0.5*logvar) # logvar becomes really negative really fast, want to avoid that
        eps = torch.normal(mean=torch.zeros_like(mu), std=1)
        x_sampled = mu + eps*sigma
        return x_sampled, z