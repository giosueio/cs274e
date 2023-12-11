import sys
sys.path.append('../')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from src.models.net import *
from src.si import *
from src.util import util as ut


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# the original interpolant was trained with mnist at time zero and svhn at time one
nsamples = 3000
testloader = ut.get_testloader_celeba_cartoon()

si = LinearInterpolant

model = UNet(
    dim = 64,
    dim_mults = (1,2,4,),
    flash_attn = True,
    channels=1,
    resnet_block_groups=4,
    attn_dim_head=32,
    attn_heads=4,
).to(device)

model.load_state_dict(torch.load(f'../models/celeba_model_lin_ot/epoch_500.pt'))

model = ut.model_counter(model)
model.eval()


mcounts_celeba =[]
mcounts_cartoon = []

with torch.no_grad():
    si = si(model, device=device)
    si.loss_type = 'velocity'
    si = make_sinsq_noisy(si)
    
    i=0
    j=0
    for b, (celeba_batch, cartoon_batch) in enumerate(testloader):

        print(celeba_batch.shape, cartoon_batch.shape)
        gen_cartoon_samples = si.sample(x_initial=celeba_batch, direction='f')
        print('cartoon samples generated')
        print(f'NFE for cartoon:', model.counter)
        # convert tensors to images
        gen_cartoon_samples = gen_cartoon_samples.cpu().squeeze().detach().numpy()
        # save images
        for sample in gen_cartoon_samples:
            plt.imsave(f'../gen_samples_ot/cartoon/{i}.png', sample, cmap='gray')
            i+=1
        mcounts_cartoon.append(model.counter)
        
        model.counter = 0
        gen_celeba_samples = si.sample(x_initial=cartoon_batch, direction='r')
        print('celeba samples generated')
        print(f'NFE for celeba:', model.counter)
        # convert tensors to images
        gen_celeba_samples = gen_celeba_samples.cpu().squeeze().detach().numpy()
        # save images
        for sample in gen_celeba_samples:
            plt.imsave(f'../gen_samples_ot/celeba/{j}.png', sample, cmap='gray')
            j+=1
        mcounts_celeba.append(model.counter)
        model.counter = 0
mean_mcounts_celeba = np.mean(mcounts_celeba)
mean_mcounts_cartoon = np.mean(mcounts_cartoon)
print(f'Mean NFE for celeba: {mean_mcounts_celeba}')
print(f'Mean NFE for cartoon: {mean_mcounts_cartoon}')
