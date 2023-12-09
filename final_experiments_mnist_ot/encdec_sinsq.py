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

classifier = LatentClassifier(z_dim=10).to(device)
classifier_state_dict = torch.load('../models/classifier_mnist.pt')
classifier.load_state_dict(classifier_state_dict)
encoder = classifier.encoder

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

si = EncoderDecoderInterpolant(model, device=device)
si = make_sinsq_noisy(si,)

optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 500
scheduler = optim.lr_scheduler.StepLR(optimizer, epochs//2, gamma=0.1)

si.train(combined_dataloader, optimizer, epochs, scheduler=scheduler, mcls=True, encoder=encoder, 
         loss_type='velocity', eval_int=50, save_int=50, save_path=f'../models/mnist_model_encdec_ot')
