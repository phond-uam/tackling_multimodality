# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:39:28 2023

@author: Michel Frising
         michel.frising@uam.es
"""
import wandb

def train_log(out, example_ct, epoch, path):
    out['epoch'] = epoch
    wandb.log(out, step=example_ct)
    
def eval_log(eval_loss, example_ct, epoch, path):
    wandb.log({"epoch": epoch,
               "eval_loss": eval_loss,
              }, step=example_ct)