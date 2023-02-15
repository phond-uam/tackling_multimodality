# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:35:47 2021

@author: Michel Frising
         michel.frising@uam.es
"""
from tqdm import tqdm
import torch
import torch.optim
import os
import numpy as np
import pickle as pickle
from configs import data_config, train_config
import argparse

parser = argparse.ArgumentParser(description='Trains either a CINN or CVAE on the slit dataset. The configuration files can be found in configs.py')
parser.add_argument('model', nargs='?', default='cinn', choices=['cinn','cvae'], help='You can train eiter a CINN or a CVAE.')
args = parser.parse_args(['cvae'])

if args.model=='cinn':
    from configs import model_config_cinn as model_config
    train_config['y_noise_scale'] = 0.01
    train_config['x_noise_scale'] = 0.0
    train_config['z_noise_scale'] = 1.0
    
elif args.model=='cvae':
    from configs import model_config_cvae as model_config
    if model_config['dim_z']%2 > 0:
        raise ValueError(f"dim_z must be divisible by 2 but is {model_config['dim_z']}")

try:
    import wandb
    wandb.login()    
    wandb_installed = True
except ImportError:
    wandb_installed = False

# check if gpu is availabe
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training {model_config['model_type']} on device {device}.")

train_config['device'] = device

#%% import data set
from data import loadData
train_loader, eval_loader, dataSet = loadData(data_config, batch_size=train_config['batch_size'], split_ratio=0.7)

#%%
from models import makeModel
model_config['dim_x'], model_config['dim_y'] = dataSet.getXDim(), dataSet.getYDim()
    
model, model_config = makeModel(model_config)
model = model.to(device)

#%% train
from utils import train_batch, eval_batch, generate_samples, histograms

# set everything up for checkpoints
from datetime import datetime
d = datetime.now()
timestamp = d.strftime('%Y-%m-%d_%H%M%S')
train_config['timestamp'] = timestamp

save_path = f"./logs/{model_config['model_type']}/{timestamp}/"
os.makedirs(save_path, exist_ok=True)
checkpt_path = f"./checkpoints/{timestamp}/"
os.makedirs(checkpt_path, exist_ok=True)
checkpoint_file = checkpt_path + "checkpoint.tar"

if wandb_installed:
    run = wandb.init(project=f"{model_config['model_type']}", config=train_config, name=timestamp)
    from wandb_utils import train_log, eval_log
else:
    # need to come up with something useful here
    from utils import train_log, eval_log
    
params = model.parameters()

optimizer = torch.optim.AdamW(params, lr=train_config['learning_rate'])
# Set up a scheduler to reduce the learning rate
step_size= 2*train_config['epochs']//3
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
model_type = model_config['model_type']
batch_ct = 0
example_ct = 0
loss_fn = lambda z, log_j: torch.mean(z**2)/2 - torch.mean(log_j)/model_config['dim_x'] 
loss_mean = []

# choose one sample at random for generating histograms 
xy_test = next(iter(eval_loader))
x_test, y_test = xy_test['x'][0], xy_test['y'][0]

x_gen = generate_samples(x_test, y_test, model, model_config, train_config, device=device)

histograms(x_gen, x_test, 0, save_path)
#%%
# ToDo: is it really necessary to have epochs and example_counts?
for epoch in range(train_config['epochs']):
    with tqdm(enumerate(train_loader), position=0, leave=True, total=len(train_loader), ascii=True, ncols=80) as pbar:
        for i, data in pbar:
            out = train_batch(data, model, optimizer, loss_fn, model_config, train_config)
            loss_mean.append(out['loss'].item())

            # housekeeping
            pbar.set_postfix({"Epoch": '{:03d}'.format(epoch), "training_loss": '{:+.3f}'.format(loss_mean[-1])})
    
            example_ct += len(data['x'])
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(out, example_ct, epoch, save_path)
        
    scheduler.step()
    # calculate validation loss every 10 epochs and do an eval step to see how
    # the network learns to approximate the dataset
    if epoch%10 == 0:
        val_loss = []
        with tqdm(enumerate(eval_loader), position=0, leave=True, total=len(eval_loader), ascii=True, ncols=80) as pbar:
            for i, data in pbar:
                loss = eval_batch(data, model, model_config, train_config)
                val_loss.append(loss.mean().item())
                # update progress bar
                pbar.set_postfix({"Epoch": '{:03d}'.format(epoch), "val_loss": '{:+.3f}'.format(val_loss[-1])})
        val_loss = np.mean(val_loss)
        eval_log(val_loss, example_ct, epoch, save_path)
        # test the generation
        x_gen, _ = generate_samples(x_test, y_test, model, model_config, train_config, device=device)
        histograms(x_gen, x_test, epoch+1, save_path)   
                
    # save checkpoint every 50 epochs
    # see also here:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    if epoch%50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, checkpoint_file)
if wandb_installed:    
    run.finish()
#%% save params for evaluating later
# need to save model weights and configuration files so I can reconstruct the 
# models

# save model state_dict
save_path = f"./trained/{model_config['model_type']}/"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), f"{save_path}{timestamp}_{model_config['model_type']}.pt")

# save model params for buiding the model for eval
save_path = f"./configs/{model_config['model_type']}/"
os.makedirs(save_path, exist_ok=True)
with open( f"{save_path}{timestamp}_params.pkl", "wb" ) as file:
    pickle.dump( (data_config, model_config, train_config), file )

# save loss, if wandb is not available save the losses to file to plot later
save_path = f"./logs/{model_config['model_type']}/"
os.makedirs(save_path, exist_ok=True)
with open( f"{save_path}{timestamp}_diagnostics.pkl", "wb" ) as file:
    pickle.dump( loss_mean, file )