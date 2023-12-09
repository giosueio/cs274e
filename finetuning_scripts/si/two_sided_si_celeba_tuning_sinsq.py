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

combined_dataloader = ut.get_doubleloader_celeba_cartoon(batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lrs = [1e-3, 1e-4, 1e-5,]
for lr in lrs:
    model = UNet(
        dim = 64,
        dim_mults = (1,2,4,),
        flash_attn = True,
        channels=1,
        resnet_block_groups=4,
        attn_dim_head=32,
        attn_heads=4,
    ).to(device)

    si = EncoderDecoderInterpolant(model, device=device)
    si = make_sinsq_noisy(si,)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    epochs = 100
    si.train(combined_dataloader, optimizer, epochs,loss_type='velocity', eval_int=5, save_int=10, save_path=f'../finetuning_scripts/celeba_demo_two_sided/encdec_sinsq_{lr}')

    del model
    del si
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    print(f'Finished {lr}')







