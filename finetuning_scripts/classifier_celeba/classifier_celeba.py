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

# Deactivate FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

mixed_dataloader, test_loader = ut.get_mixedloader_celeba_cartoon(batch_size=32, return_testloader=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

betas = [0.01,0.005]
lrs = [1e-4, ]

best_accuracy = 10e5
best_loss = 10e5

save_int = 10


for b in betas:
    for lr in lrs:
        classifier = BigLatentClassifier(z_dim=10).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        save_path = f"../finetuning_scripts/classifier_celeba/celeba_beta_{b}_lr_{lr}.pt"

        classifier.train()
        
        for epoch in tqdm(range(100)):
            for i, batch in enumerate(mixed_dataloader):
                optimizer.zero_grad()
                batch_size = batch[0].shape[0]
                x, y = batch
                x = x.to(device)
                y = y.to(device).float()
                logits, z_mu, z_sigma = classifier(x)

                l = ut.compute_loss_BLC(logits, z_mu, z_sigma, y, beta=b)
                l.backward()
                optimizer.step()
            classifier.eval()
            test_loss = 0
            correct = 0.00
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device).float()

                    logits, z_mu, z_sigma = classifier(x)
                    test_loss += ut.compute_loss_BLC(logits, z_mu, z_sigma, y, beta=b).item()
                    prediction = torch.stack([i>0 for i in logits]).float()
                    is_correct = prediction == y
                    num_correct = torch.sum(is_correct)
                    correct += num_correct
            if epoch % save_int == 0:
                save_path_epoch = save_path[:-3] + f"_{epoch}.pt"
                torch.save(classifier.state_dict(), save_path_epoch)

            test_loss /= len(test_loader.dataset)
            correct /= 2*len(test_loader.dataset)
            print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct:.4f}')
        print(f"beta: {b}, lr: {lr}")
        print()

        if test_loss < best_loss:
            best_loss = test_loss
            best_accuracy = 100. * correct / len(test_loader.dataset)
            torch.save(classifier.state_dict(), )