# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:08:37 2020

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
import numpy as np
import pickle 
import os 
import pywt

class SlitDataset(torch.utils.data.Dataset):
    def __init__(self, data_config):
        print("Loading transmission spectra...")
        with open(os.path.join(".", "data", data_config['data_file']), "rb") as file:
            (w, params, spectra) = pickle.load(file)

        self.wavelengths = w
        self.param_labels = ['p1', 'h1', 'p2', 'h2']
        preprocessing = data_config['processing']
        self.preprocessing = preprocessing
        # apply data preprocessing
        if preprocessing['log']:
            print("Applying log transform...")
            spectra = np.log(spectra)
        
        if preprocessing['wavelet_dec']:
            print("Applying wavelet decomposition...")
            spectra = pywt.wavedec(spectra, 'rbio3.1')
            split_indices = [s.shape[-1] for s in spectra]
            self.split_indices = np.cumsum(split_indices)[:-1]
            spectra = np.concatenate(spectra, axis=-1)
        
        if preprocessing['normalize']:
            print("Applying normalization...")
            self.means_params = params.mean(axis=0)
            self.stds_params  = params.std(axis=0)
            params = (params-self.means_params)/self.stds_params
            self.means_spectra = spectra.mean(axis=0)
            self.stds_spectra  = spectra.std(axis=0)
            spectra = (spectra-self.means_spectra)/self.stds_spectra
        
        # all preprocessing done, convert to torch tensor
        self.spectra = torch.FloatTensor(spectra)
        self.params = torch.FloatTensor(params)
        self.n_samples = params.shape[0]

        # if provided, load a forward model to generate data
        self.forwardModel = None
        if data_config['forwardModel'] is not None:
            model = torch.load(os.path.join(".","data","dataGeneratingModel",data_config['forwardModel']), map_location=torch.device(type='cpu'))
            model.eval()
            self.forwardModel = model

        print("Done loading data...")
   
    def undoXPreprocessing(self, x):
        if self.preprocessing['normalize']:
            x = x*self.stds_params + self.means_params
        return x.float() # keep it consistent, the numpy arrays are doubles
    
    def undoYPreprocessing(self, y):
        if self.preprocessing['normalize']:
            y = y*self.stds_spectra + self.means_spectra
        if self.preprocessing['wavelet_dec']:
            y = pywt.waverec(np.split(y, self.split_indices, axis=-1), 'rbio3.1')
        if self.preprocessing['log']:
            y = np.exp(y)
        return y     
    
    @torch.no_grad()
    def predict(self, p):
        # This particular model takes normalized x as input and outputs
        # unnormalized y
        if self.forwardModel is not None:
            return self.forwardModel(p)
        else:
            raise AttributeError("Data generation not available")
      
    def __len__(self):
        return self.n_samples # 2000 # 

    def getYDim(self):
        return self.spectra.shape[1]
    
    def getXDim(self):
        return self.params.shape[1]
    
    def __getitem__(self, idx):
        x = self.params[idx]
        y = self.spectra[idx]
        sample = {'x': x,
                  'y': y,
                  }
        return sample