import sys
sys.path.append('../')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from src.models.net import *
from src.si import *
from src.util import util as ut

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

combined_dataloader = ut.get_doubleloader_mnist_and_svhn_data(batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 1e-3

model = UNet(
    dim = 28,
    dim_mults = (1,2,),
    flash_attn = True,
    channels=1,
    resnet_block_groups=4,
    attn_dim_head=32,
    attn_heads=4,
).to(device)

si = LinearInterpolant(model, device=device)
si = make_sinsq_noisy(si,)

optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 500
scheduler = optim.lr_scheduler.PolynomialLR(optimizer, [lr, lr/10], epochs)

si.train(combined_dataloader, optimizer, epochs,loss_type='velocity', scheduler=scheduler,
         eval_int=50, save_int=50, save_path=f'../finetuning_scripts/mnist_model_lin')