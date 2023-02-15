# How to use this code
The code has few requirements, which can be found in the `requirments.txt` file. The use of wandb is optional. If wandb is not installed, plots are generated with matplotlib, tracking the progress.

# 1. Download the data
The dataset needed for training is distributed separtedly and can be downloaded [here](https://drive.google.com/file/d/1Z1yH2B0YC13l6w2tYq9HcAvyf9qyLDLI/view?usp=share_link) and needs to be placed in the `data` folder. The datasets consists of parameters describing the slit flanked by periodic gratings and the corresponding transmission spectra. The data is packaged as a binary file and can be loaded with `pickle`. And example how to do that canbe found in the `dataset.py` file in the `data` folder.

# 2. Training
After downloading the data in step 1. and placing it into the `data` folder the network can be trained by running.
```
python train.py
```
The code automatically generates diagnostics that are visualized with `wandb`, if installed. If `wandb` is not available, diagnostic plots are generated with matplotlib during training. At the end of the training weights and configurations are automatically saved to disk. The model weights are also saved every 50 epochs. Pretrained model weights are also provided in the output folder.

# 3. Evaluation
Pretrained model weights can be found in the output folder. The evaluation consists of checking the predicted errors of the generated data compared to the target. A second test consists of selecting one transmission spectrum and generating the histograms of the parameters corresponding to that transmission spectrum, effectively sampling the posterior 