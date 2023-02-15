# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:35:02 2023

@author: Michel Frising
         michel.frising@uam.es
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":16,
    })
matplotlib.use('AGG')
plt.ioff()

conv_models = ['resNet']

# Problem: train_batch is different for cinn and cvae, see how I solve that problem
def train_batch(data, model, optimizer, loss_fn, model_config, train_config):
    x, y = data['x'], data['y']
    # print(x.size(), y.size())
    x, y = x.to(device=train_config['device']), y.to(device=train_config['device'])
    
    # add some noise to y
    if model_config['model_type'] == "cinn":
        if model_config['cond_net'] in conv_models:
            y = y.unsqueeze(1)
        if train_config['y_noise_scale'] > 0:
            y += train_config['y_noise_scale']*(1-2*torch.rand_like(y))
        # send the batch through the model
        out = model(x, y)
        # apply loss fcn
        loss = loss_fn(out['z'], out['log_jac_det']) # ToDo: move that inside the function
        out['loss'] = loss
    elif model_config['model_type'] == "cvae":
        # send the batch through the model
        out = model(x, y)
        # apply loss fcn
        loss = out['loss']
    # calculate the gradients and apply backpropagation
    optimizer.zero_grad()
    loss.backward()
    # might be necessary to clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.02) # the gradient clip value probably needs to be optimized
    optimizer.step()
    
    return out

# the loss function of the cinn checks how close the latent space is to a Gaussian
# maybe for reconstruction it is more informative to see how well the test set
# is reconstructed?
# Is it really useful to do this while training?
# Usually for eval I choose one sample and expand to get spread
# Maybe doing that and saving the scatter plot is more useful

# ToDo: switching to dicts for output breaks a few things
def eval_batch(data, model, model_config, train_config):
    x, y = data['x'], data['y']
    # print(x.size(), y.size())
    x, y = x.to(device=train_config['device']), y.to(device=train_config['device'])
    # add some noise to y
    if model_config['model_type'] == "cinn":
        if model_config['cond_net'] in conv_models:
            y = y.unsqueeze(1)
        if train_config['y_noise_scale'] > 0:
            y += train_config['y_noise_scale']*(1-2*torch.rand_like(y))
        # sample the latent space
        z = train_config['z_noise_scale']*torch.rand_like(x).to(device=train_config['device'])
        x_gen, _ = model.reverse_sample(z, y, jac=False)
    elif model_config['model_type'] == "cvae":
        x_gen, z = model.sample(x, y)
    # calculating the L2 norm
    diff = torch.sqrt(torch.sum(torch.square(x-x_gen), 1))
    
    return diff

def generate_samples(x, y, model, model_config, train_config, n_samples=int(1e4), device=torch.device('cpu')):
    x, y = x.to(device=device), y.to(device=device)
    y = y.expand(n_samples,-1)
    if model_config['model_type'] == "cinn":
        if model_config['cond_net'] in conv_models:
            y = y.unsqueeze(1)
        z_noise_scale = train_config['z_noise_scale']
        dim_x = x.shape[-1]
        with torch.no_grad():
            z = z_noise_scale*torch.randn(n_samples, dim_x, device=device)
            x_gen, _ = model.reverse_sample(z, y, jac=False)
    elif model_config['model_type'] == "cvae":
        x = x.expand(n_samples,-1)
        x_gen, z = model.sample(x, y)

    return x_gen, z
    
def train_log(train_loss, example_ct, epoch, path):
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(epoch), train_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train_loss')  
    fig.set_tight_layout(True)
    fig.savefig(f"{path}train_loss.png")
    
def eval_log(eval_loss, example_ct, epoch, path):
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(epoch), eval_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('eval_loss')  
    fig.set_tight_layout(True)
    fig.savefig(f"{path}eval_loss.png")

# ToDo: this needs to be focused on one sample
def histograms(x_gen, x_ref, epoch=-1, path=None):
    plt.close('all')
    x_gen, x_ref = x_gen.cpu().numpy(), x_ref.cpu().numpy()
    n_samples = x_gen.shape[0]
    fig, axs = plt.subplots(2,2)
    axs = axs.reshape(-1)
    x_ref_sym = np.roll(x_ref,2)
    for i, ax in enumerate(axs):
        ax.hist( x_gen[:,i], bins=int(np.sqrt(n_samples)) )
        y_lims = ax.get_ylim()
        ax.plot(2*[x_ref[i]],y_lims,'k--')
        ax.plot(2*[x_ref_sym[i]],y_lims,'k--')
        ax.set_xlabel(f"$p_{i}$")
        ax.set_xlim([-2,2])
        ax.set_ylabel("counts")
    fig.set_tight_layout(True)
    if path is not None:
        fig.savefig(f"{path}hist_{epoch}.png")

def cluster(x_gen, x_ref):
    cluster_centers = np.c_[ x_ref, np.r_[x_ref[2:],x_ref[:2]] ].T
    kmeans = KMeans(n_clusters=2, init=cluster_centers).fit(x_gen)
    cluster_centers = kmeans.cluster_centers_
    indices = kmeans.labels_.astype(bool)
    
    return cluster_centers, indices

def scatterplot(x_gen, x_ref, s=0.1, clustering=False):
    x_gen, x_ref = x_gen.numpy(), x_ref.numpy()
    
    if clustering:
        cluster_centers, indices = cluster(x_gen, x_ref)
    else:
        cluster_centers, indices = None, None
    
    # only plot the periodicities
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    # turn off labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    if clustering:
        for ind, cc in zip([indices, ~indices], cluster_centers):
            ax_scatter.scatter(x_gen[ind,0], x_gen[ind,2],s)
            ax_scatter.plot(cc[0], cc[2],'ro')
            ax_scatter.plot(cc[0], cc[2],'ro')
            ax_histx.hist(x_gen[ind,0], bins=100)
            ax_histy.hist(x_gen[ind,2], bins=100, orientation='horizontal')
    else:
        ax_scatter.scatter(x_gen[:,0], x_gen[:,2],s)
        ax_histx.hist(x_gen[:,0], bins=100)
        ax_histy.hist(x_gen[:,0], bins=100, orientation='horizontal')
    # plot the reference
    ax_scatter.plot(x_ref[0], x_ref[2],'kx')
    ax_scatter.plot([0,x_ref[2],x_ref[2]],[x_ref[0],x_ref[0],0],'k--')
    ax_scatter.plot([0,x_ref[0],x_ref[0]],[x_ref[2],x_ref[2],0],'k--')
    # make sure the axes are nice
    ax_scatter.set_xlim(ax_histx.get_xlim())
    ax_scatter.set_ylim(ax_histy.get_ylim())
    
    return fig

def plot_latent_space(z, clustering=False, x_gen=None, x_ref=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    axs = np.empty((4,4), dtype=object)
    for i in range(4):
        for j in range(i,4):
            axs[i,j] = fig.add_subplot(gs[i,j])
    ax_scatter = fig.add_subplot(gs[2:,:2])
    
    sigmas = [1,3]
    if clustering:
        _, indices = cluster(x_gen, x_ref)
    else:
        indices = None
    
    if clustering:
        for ind in [indices, ~indices]:
            ax_scatter.scatter(x_gen[ind,0], x_gen[ind,2],0.5)
            ax_scatter.set_xlabel("$p_1$")
            ax_scatter.set_ylabel("$p_2$")
            for i in range(4):
                for j in range(i,4):
                    if i==j:
                        axs[i,i].hist(z[ind,i], bins=50)
                    else:
                        axs[i,j].scatter(z[ind,i], z[ind,j],0.1)                
    else:
        ax_scatter.scatter(x_gen[ind,0], x_gen[ind,2],0.5)
        for i in range(4):
            for j in range(i,4):
                if i==j:
                    axs[i,i].hist(z[:,i], bins=50)
                else:
                    axs[i,j].scatter(z[:,i], z[:,j],0.1)    
    # make it pretty         
    for i in range(4):
        for j in range(i,4):
            if i==j:
                axs[i,i].set_xlabel(f"$z_{i}$")
                axs[i,i].set_xlim([-4,4])
                axs[i,i].set_ylim([0,335])
                axs[i,i].set_aspect(1./axs[i,i].get_data_ratio())
            else:
                for s in sigmas:
                    circ = Circle((0,0), s, fill=None)
                    axs[i,j].add_artist(circ)
                axs[i,j].set_xlabel(f"$z_{i}$")
                axs[i,j].set_ylabel(f"$z_{j}$")
                axs[i,j].set_xlim([-4,4])
                axs[i,j].set_ylim([-4,4])
                axs[i,j].set_aspect('equal')
                