# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:46:38 2021

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
from torch.utils.data import DataLoader, random_split
from .dataSet import SlitDataset

def loadData(data_config, batch_size, split_ratio=0.7):
    dataSet = SlitDataset(data_config)
    
    # split data into training data and test data
    n_train = int(split_ratio*dataSet.__len__())
    n_test  = dataSet.__len__()-n_train
    data_train, data_test = random_split(dataSet, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    # Using random_split with a generator with fixed seed makes the dats split reproducible
    # which in turn is important for the evaluation part
    print(f"{n_train} samples for training, \n {n_test} samples for validation\n({n_train+n_test} samples in total)")
    
    train_loader = DataLoader(data_train, batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=True)
    
    test_loader = DataLoader(data_test, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    
    return train_loader, test_loader, dataSet