# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:08:13 2023

@author: Michel Frising
         michel.frising@uam.es
"""
import torch.nn as nn

data_config = dict(data_set = "slits",
                   data_file = "slitData.pkl",
                   forwardModel = "dense.pt",
                   processing = dict(log=False,
                                     wavelet_dec=True,
                                     normalize=True,
                                     ),
                   )    

# see the corresponding files in the model folder for the hyperparameters of the networks
# if set to blank, standard values are used, see makeModel what they are
model_config_cinn = dict(model_type='cinn', # options are 'cinn', 'cvae'
                    cinn_params=dict(dense_neurons=512, # number of neurons in the hidden layer of the coupling layer
                                      N_coupling_blocks=6),
                    cond_net='resNet', # options are 'dense', 'resNet'
                    cond_params=dict(block='BasicBlock', #<--ToDo: change that in the Resnet architecture
                                     layers=[2, 2, 2, 2],
                                     out_dim=512,),
                    )

model_config_cvae = dict(model_type='cvae',
                    encoder_config=dict(hidden_layers=5*[256],
                                        activation='relu',
                                        output_activation=None,
                                        bias=False,
                                        init='uniform',
                                        use_batchnorm=True,
                                        use_weightnorm=True,# seems to be necessary to stabilize training
                                        ),
                    decoder_config=dict(hidden_layers=5*[256],
                                        activation='relu',
                                        output_activation=None,
                                        bias=False,
                                        init='uniform',
                                        use_batchnorm=True,
                                        use_weightnorm=True,
                                        ),
                    dim_z=4,
                    alpha=0.85,
                    )

train_config = dict(epochs=300,
                       batch_size=128,
                       learning_rate=1e-3,
                   )