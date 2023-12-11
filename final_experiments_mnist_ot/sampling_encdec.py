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
mnist_testloader = ut.get_mnist_test()
mnist_3000_testsamples = torch.stack([mnist_testloader.dataset[i][0] for i in range(nsamples)]).to(device)
svhn_testloader = ut.get_svhn_test()
svhn_3000_testsamples = torch.stack([svhn_testloader.dataset[i][0] for i in range(nsamples)]).to(device)

# save real samples
svhn_3000_testsamples_np = svhn_3000_testsamples.squeeze().cpu().numpy()
mnist_3000_testsamples_np = mnist_3000_testsamples.squeeze().cpu().numpy()

model_names = ['encdec', 'lin']
SIs = [EncoderDecoderInterpolant, LinearInterpolant, LinearInterpolant, 
       partial(PolynomialInterpolant, p=0.5), partial(PolynomialInterpolant, p=2)]


for m in range(2):
    mcounts_svhn = []
    mcounts_mnist = []  
    model = UNet(
        dim = 28,
        dim_mults = (1,2,),
        flash_attn = True,
        channels=1,
        resnet_block_groups=4,
        attn_dim_head=32,
        attn_heads=4,
    ).to(device)

    model.load_state_dict(torch.load(f'../models/mnist_model_{model_names[m]}_ot/epoch_500.pt'))
    
    model = ut.model_counter(model)
    model.eval()

    with torch.no_grad():
        si = SIs[m](model, device=device)
        si.loss_type = 'velocity'
        if model_names[m] == 'linsb':
            si = make_noisy(si, 0.5)
        else:
            si = make_sinsq_noisy(si)
        
        i=0
        j=0
        for b in tqdm(range(100)):
            x_initial_mnist = mnist_3000_testsamples[b*30:(b+1)*30]
            gen_svhn_samples = si.sample(x_initial=x_initial_mnist, direction='f')
            print('svhn samples generated')
            print(f'NFE for svhn for model {model_names[m]}:', model.counter)
            # convert tensors to images
            gen_svhn_samples = gen_svhn_samples.cpu().squeeze().detach().numpy()
            # save images
            for sample in gen_svhn_samples:
                plt.imsave(f'../gen_samples_ot/{model_names[m]}/svhn/{i}.png', sample, cmap='gray')
                i+=1
            mcounts_svhn.append(model.counter)
            
            model.counter = 0
            x_initial_svhn = svhn_3000_testsamples[b*30:(b+1)*30]
            gen_mnist_samples = si.sample(x_initial=x_initial_svhn, direction='r')
            print('mnist samples generated')
            print(f'NFE for mnist for model {model_names[m]}:', model.counter)
            # convert tensors to images
            gen_mnist_samples = gen_mnist_samples.cpu().squeeze().detach().numpy()
            # save images
            for sample in gen_mnist_samples:
                plt.imsave(f'../gen_samples_ot/{model_names[m]}/mnist/{j}.png', sample, cmap='gray')
                j+=1
            mcounts_mnist.append(model.counter)
            model.counter = 0
    mean_mcounts_svhn = np.mean(mcounts_svhn)
    mean_mcounts_mnist = np.mean(mcounts_mnist)
    print(f'Mean NFE for svhn: {mean_mcounts_svhn}')
    print(f'Mean NFE for mnist: {mean_mcounts_mnist}')
