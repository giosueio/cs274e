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

mixed_dataloader = ut.get_mnist_and_svhn_data(batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
test_loader = ut.get_test_data()

betas = [0.01, 0.05, 0.1, 0.5, 1.0]
lrs = [1e-4, 5e-4, 1e-3, 5e-3]

best_accuracy = 10e5
best_loss = 10e5

for b in betas:
    for lr in lrs:
        classifier = LatentClassifier(z_dim=10).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)

        classifier.train()
        for epoch in tqdm(range(100)):
            for i, batch in enumerate(mixed_dataloader):
                optimizer.zero_grad()

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                logits, z_mu, z_sigma = classifier(x)
                
                l = ut.compute_loss_LC(logits, z_mu, z_sigma, y, beta=b)
                l.backward()
                optimizer.step()
        
        classifier.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                logits, z_mu, z_sigma = classifier(x)
                test_loss += ut.compute_loss_LC(logits, z_mu, z_sigma, y, beta=.05).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print(f"beta: {b}, lr: {lr}")
        print()

        if test_loss < best_loss:
            best_loss = test_loss
            best_accuracy = 100. * correct / len(test_loader.dataset)
            torch.save(classifier.state_dict(), f"best_classifier_beta_{b}_lr_{lr}.pt")



        



