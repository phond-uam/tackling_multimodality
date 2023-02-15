# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:46:54 2023

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt

from utils import (eval_batch, generate_samples, histograms_eval,
                   scatterplot, plot_latent_space)

import matplotlib # Not sure if that is necessary
matplotlib.use('QtAgg')
plt.ion()

conv_models = ['resNet']

# Define some common parameters
parser = argparse.ArgumentParser(description='Trains either a CINN or CVAE on the slit dataset. The configuration files can be found in configs.py')
parser.add_argument('model', help='You can train eiter a CINN or a CVAE.')
parser.add_argument('model_to_eval', help='Here goes the timestamp of the model you want to evaluate')
args = parser.parse_args(['cinn', '2023-02-08_161035'])
# args = parser.parse_args(['cvae', '2023-02-08_112832'])

model_type = args.model
model_to_eval = args.model_to_eval 

# ToDo: What is actually the folder structure? Right now I just store everything
# in output
with open( f"./configs/{model_type}/{model_to_eval}_params.pkl", "rb" ) as file:
    data_config, model_config, train_config = pickle.load( file )

device = "cpu" # eval on cpu
#%% Load data for evaluation
from data import loadData
train_loader, eval_loader, dataSet = loadData(data_config, batch_size=train_config['batch_size'], split_ratio=0.7)

#%% Build the model
from models import makeModel

model, model_config = makeModel(model_config)
model = model.to(device)

#%% Load weights from file ()
# ToDo: model weights should be saved in pretrained or trained folder
state_dict = {k:v for k,v in torch.load(f"./trained/{model_type}/{model_to_eval}_{model_type}.pt", map_location=torch.device(device)).items() if 'tmp_var' not in k}
model.load_state_dict(state_dict)

model.eval()

#%% 1) error on the reconstructed data
diff_train = np.empty((0,))
diff_eval = np.empty((0,))

z_noise_scale = 1.0
with torch.no_grad():
    print("Evaluating the training set")
    for sample in tqdm(train_loader):
        diff = eval_batch(sample, model, model_config, train_config)
        diff_train = np.r_[diff_train, diff.numpy()]
    print("Evaluating the evaluation set")
    for sample in tqdm(eval_loader):
        diff = eval_batch(sample, model, model_config, train_config)
        diff_eval = np.r_[diff_eval, diff.numpy()]
# for some reason some of the values are inf, we need to take care of them
plt.close('all')
_, axs = plt.subplots(1,2)
labels = ['training set', 'evaluation set']
for ax, data, label in zip(axs, [diff_train, diff_eval], labels):
    quantiles = np.quantile(data, [0.5, 0.95])
    indices = np.where(data<quantiles[1])
    ax.hist( data[indices], bins=int(np.sqrt(len(data[indices]))), label='residues' )
    ax.set_xlabel('residues')
    ax.set_ylabel('counts')
    ylims = ax.get_ylim()
    ax.plot( [quantiles[0], quantiles[0]], ylims, 'k--', label='median' )
    ax.set_title(label)
    ax.legend()

#%% 2) take samples from the eval set and generate devices, regenerate the spectra and plot 
from utils import (eval_batch, generate_samples, histograms,
                    scatterplot, plot_latent_space)

matplotlib.use('QtAgg')
plt.ion()

n_examples = 1 # don't choose this too high, otherwise it is going to take ages
n_samples = int(1e4) # dito here

eval_dataset = eval_loader.dataset
# _, axs = plt.subplots(2,2)
# axs = axs.reshape(-1)
plt.close('all')
for i in range(n_examples):
    # generate samples
    idx = np.random.randint(len(eval_dataset))
    sample = eval_dataset[idx]
    x, y = sample['x'], sample['y']
    # y_pred = dataSet.predict(x.expand(1,-1)).numpy()
    # axs[i].plot(dataSet.undoYPreprocessing(y.numpy()))
    # axs[i].plot(y_pred.squeeze(),'--')
    x_gen, z = generate_samples(x, y, model, model_config, train_config, n_samples)
    # histograms of the generated device params
    histograms(x_gen, x)
    # scatter plots of p1 and p2
    scatterplot(x_gen, x, s=0.1, clustering=True)
    # reconstruct spectra
    y_gen = dataSet.predict(x_gen)
    y = torch.FloatTensor(dataSet.undoYPreprocessing(y.unsqueeze(0).numpy())) 
    diff = torch.sqrt(torch.sum(torch.square(y[:,:-1]-y_gen), 1)).numpy() # ToDo find a better way to undo the normalization
    _, ax = plt.subplots(1,1)
    cutoff = np.quantile(diff,[0.98])
    indices = diff<cutoff
    ax.hist(diff[indices], bins=int(np.sqrt(n_samples)))
    ax.set_xlabel('residues')
    ax.set_ylabel('counts')
    ax.set_title('reconstructed spectra')
    # now let us have a loot at latent space
    plot_latent_space(z, clustering=True, x_gen=x_gen, x_ref=x)
    plt.show()
