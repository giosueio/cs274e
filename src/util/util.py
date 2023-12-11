import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
import tarfile
import os
from PIL import Image
import pandas as pd



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.csv_paths = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
        self.labels = self.load_labels()

    def load_labels(self):
        labels = {}
        for csv_path in self.csv_paths:
            df = pd.read_csv(os.path.join(self.root_dir, csv_path))
            filename = f'{csv_path[:-3]}png'
            labels[filename] = []
            for rnum, row in df.iterrows():
                if rnum == 7:
                    labels[filename].append(row[1]==14)
                elif rnum == 12:
                    labels[filename].append(row[1]==11)
            labels[filename] = torch.tensor(labels[filename])
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label = self.labels[img_name]  # Adjust this line based on your CSV structure

        if self.transform:
            image = self.transform(image)

        return image, label


class CombinedDataset(Dataset):
    '''
    Combines two datasets of the same length into one. Used for building interpolant from one dataset to another.
    '''

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(dataset1) == len(dataset2), "Datasets must have the same length."

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        return [self.dataset1[index], self.dataset2[index]]

def plot_loss_curve(tr_loss, save_path, te_loss=None, te_epochs=None, logscale=True):
    fig, ax = plt.subplots()

    if logscale:
        ax.semilogy(tr_loss, label='tr')
    else:
        ax.plot(tr_loss, label='tr')
    if te_loss is not None:
        te_epochs = np.asarray(te_epochs)
        if logscale:
            ax.semilogy(te_epochs-1, te_loss, label='te')  # assume te_epochs is 1-indexed
        else:
            ax.plot(te_epochs-1, te_loss, label='te')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close(fig)


def plot_samples(samples, save_path):
    n = samples.shape[0]
    sqrt_n = int(np.sqrt(n))

    fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(8,8))

    samples = samples.permute(0, 2, 3, 1)  # (b, c, h, w) --> (b, h, w, c)
    samples = samples.detach().cpu()

    for i in range(n):
        j, k = i//sqrt_n, i%sqrt_n
        
        axs[j, k].imshow(samples[i])
        
        axs[j, k].set_xticks([])
        axs[j, k].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(save_path)
    plt.close(fig)

def get_unlabeled_mnist_data():
    class MNISTImagesOnly(datasets.MNIST):
        def __getitem__(self, index):
            img, _ = super().__getitem__(index)
            return img

    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        MNISTImagesOnly('../data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTImagesOnly('../data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )
    return train_loader, test_loader

def get_svhn_data():
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', split='extra', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    return train_loader, (None, None), (None, None)

def get_transformed_svhn_data():
    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', split='extra', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    return train_loader

def get_mnist_and_svhn_data(batch_size=100):
    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])
        
    d1 = datasets.MNIST('../data', train=True, download=True, transform=preprocess)
    d2 = datasets.SVHN('../data', split='extra', download=True, transform=preprocess)

    d2_reduced = torch.utils.data.Subset(d2, np.random.choice(len(d2), len(d1), replace=False))

    dataset = torch.utils.data.ConcatDataset([d1, d2_reduced])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    
    return train_loader

def get_doubleloader_mnist_and_svhn_data(batch_size=100, 
                                         seed=0,
                                         augment_mnist=False,
                                         return_testloader=False,):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])
    
    if augment_mnist:
        preprocess_mnist = transforms.Compose([
                    preprocess,
                    transforms.RandomInvert(p=0.5),
        ])
    else:
        preprocess_mnist = preprocess    
        
    d1 = datasets.MNIST('../data', train=True, download=True, transform=preprocess_mnist)
    d2 = datasets.SVHN('../data', split='extra', download=True, transform=preprocess)

    train_indices = np.random.choice(len(d2), len(d1), replace=False)
    d2_reduced = torch.utils.data.Subset(d2, train_indices)

    

    d1_nolabels = [data[0] for data in d1]
    d2_nolabels = [data[0] for data in d2_reduced]

    dataset = CombinedDataset(d1_nolabels, d2_nolabels)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    
    if return_testloader:
        d1_test = datasets.MNIST('../data', train=False, download=True, transform=preprocess_mnist)
        test_indices = np.setdiff1d(np.arange(len(d2)), train_indices)
        d2_test = torch.utils.data.Subset(d2, test_indices)
        d1_nolabels_test = [data[0] for data in d1_test]
        d2_nolabels_test = [data[0] for data in d2_test]
        test_dataset = CombinedDataset(d1_nolabels_test, d2_nolabels_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)
        return train_loader, test_loader
    else:
        return train_loader

def get_test_data(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])

    d1 = datasets.SVHN('../data', split='test', download=True, transform=preprocess)
    d2 = datasets.MNIST('../data', train=False, download=True, transform=preprocess)

    dataset = torch.utils.data.ConcatDataset([d1, d2])
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=True
    )
    return test_loader

def get_svhn_test(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', split='test', download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )
    return test_loader

def get_mnist_test(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )
    return test_loader

def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    # Create supervised subset (deterministically chosen)
    # This subset will serve dual purpose of log-likelihood evaluation and
    # semi-supervised learning. Pretty hacky. Don't judge :<
    X = X_test if use_test_subset else X_train
    y = y_test if use_test_subset else y_train

    xl, yl = [], []
    for i in range(10):
        idx = y == i
        idx_choice = get_mnist_index(i, test=use_test_subset)
        xl += [X[idx][idx_choice]]
        yl += [y[idx][idx_choice]]
    xl = torch.cat(xl).to(device)
    yl = torch.cat(yl).to(device)
    yl = F.one_hot(yl, num_classes=10) #yl.new(np.eye(10)[yl])
    labeled_subset = (xl, yl)

    return train_loader, labeled_subset, (X_test, y_test)

def get_mnist_index(i, test=True):
    # Obviously *hand*-coded
    train_idx = np.array([[2732,2607,1653,3264,4931,4859,5827,1033,4373,5874],
                          [5924,3468,6458,705,2599,2135,2222,2897,1701,537],
                          [2893,2163,5072,4851,2046,1871,2496,99,2008,755],
                          [797,659,3219,423,3337,2745,4735,544,714,2292],
                          [151,2723,3531,2930,1207,802,2176,2176,1956,3622],
                          [3560,756,4369,4484,1641,3114,4984,4353,4071,4009],
                          [2105,3942,3191,430,4187,2446,2659,1589,2956,2681],
                          [4180,2251,4420,4870,1071,4735,6132,5251,5068,1204],
                          [3918,1167,1684,3299,2767,2957,4469,560,5425,1605],
                          [5795,1472,3678,256,3762,5412,1954,816,2435,1634]])

    test_idx = np.array([[684,559,629,192,835,763,707,359,9,723],
                         [277,599,1094,600,314,705,551,87,174,849],
                         [537,845,72,777,115,976,755,448,850,99],
                         [984,177,755,797,659,147,910,423,288,961],
                         [265,697,639,544,543,714,244,151,675,510],
                         [459,882,183,28,802,128,128,53,550,488],
                         [756,273,335,388,617,42,442,543,888,257],
                         [57,291,779,430,91,398,611,908,633,84],
                         [203,324,774,964,47,639,131,972,868,180],
                         [1000,846,143,660,227,954,791,719,909,373]])

    if test:
        return test_idx[i]

    else:
        return train_idx[i]

def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def compute_loss_LC(logits, z_mu, z_sigma, y, beta=1.0):
    '''
    Compute the loss for the latent classifier.
    '''
    log_likelihood = F.cross_entropy(logits, y, reduction='none')
    kl = kl_normal(z_mu, z_sigma, torch.zeros_like(z_mu), torch.ones_like(z_sigma))
    loss = (log_likelihood + beta*kl).mean()
    return loss

    
def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def gaussian_wd(m1, s1, m2, s2):
    """
    Computes the Wasserstein distance between two Gaussians
    """
    return (m1 - m2).norm(2, dim=1) + (s1.sqrt() - s2.sqrt()).pow(2).sum(dim=1)

def batch_gaussian_wd(m1, s1, m2, s2):
    """
    Computes the Wasserstein distance between two batches of Gaussians
    """
    mean_sq_norm = (m1.unsqueeze(1) - m2.unsqueeze(0)).norm(2, dim=2)
    stddev_froeb_norm = (s1.unsqueeze(1) - s2.unsqueeze(0)).pow(2).sum(dim=2)
    return mean_sq_norm + stddev_froeb_norm

def optimal_WD_matching(x0, x1, encoder, return_values=False):
    h0 = encoder(x0)
    h1 = encoder(x1)
    z_mu0, z_sigma0 = gaussian_parameters(h0)
    z_mu1, z_sigma1 = gaussian_parameters(h1)
    wds = batch_gaussian_wd(z_mu0, z_sigma0, z_mu1, z_sigma1)
    indices = linear_sum_assignment(wds.cpu().detach().numpy())[1]
    x1_opti = x1[indices]
    if return_values:
        values = []
        for i in range(len(indices)):
            values.append(wds[i, indices[i]].item())
        return x1_opti, values
    else:
        return x1_opti


def celeba_dataset():
    preprocess = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.,), (1.,))
    ])

    celeba = datasets.CelebA(root='../data', split='train', download=True, transform=preprocess)
    # take first 10000 images

    # celeba_loader = torch.utils.data.DataLoader(celeba, batch_size=64, shuffle=True)

    # keep only label 15 and 24 from celeba
    celeba_labels = celeba.attr[:, (15, 24)]
    celeba.attr = celeba_labels
    return celeba

def cartoon_dataset(path_to_tgz = '../data/cartoonset10k.tgz', extraction_path = '../data'):
    # Replace '/path/to/cartoonset_dataset.tgz' with the actual path to your dataset file
    dataset_path = path_to_tgz
    extraction_path = extraction_path

    # Create the extraction directory if it doesn't exist
    os.makedirs(extraction_path, exist_ok=True)

    # Extract the .tgz file
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall(path=extraction_path)
    # Define the data transformations if needed
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Adjust size as needed
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    dataset_path = f'{extraction_path}/cartoonset10k'

    cartoon = CustomDataset(root_dir=dataset_path, transform=transform)
    return cartoon

def get_doubleloader_celeba_cartoon(batch_size=100, 
                                    seed=0,
                                    path_to_tgz = '../data/cartoonset10k.tgz', 
                                    extraction_path = '../data',
                                    return_datasets=False,
                                    return_testloader=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    celeba = celeba_dataset()
    cartoon = cartoon_dataset(path_to_tgz, extraction_path)
    print('Datasets loaded')

    cartoon_indices = np.random.choice(len(cartoon), 9000, replace=False)
    cartoon_reduced = torch.utils.data.Subset(cartoon, cartoon_indices)

    celeba_indices = np.random.choice(len(celeba), 9000, replace=False)
    celeba_reduced = torch.utils.data.Subset(celeba, celeba_indices)

    celeba_nolabels = [data[0] for data in celeba_reduced]
    cartoon_nolabels = [data[0] for data in cartoon_reduced]

    dataset = CombinedDataset(celeba_nolabels, cartoon_nolabels)
    print('Combined')
    doubleloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    if return_datasets:
        return doubleloader, celeba_reduced, cartoon
    elif return_testloader:
        celeba_test_indices = np.setdiff1d(np.arange(len(celeba)), celeba_indices)[:1000]
        celeba_test = torch.utils.data.Subset(celeba, celeba_test_indices)
        cartoon_test_indices = np.setdiff1d(np.arange(len(cartoon)), cartoon_indices)[:1000]
        cartoon_test = torch.utils.data.Subset(cartoon, cartoon_test_indices)
        test_dataset = CombinedDataset([cartoon_test, celeba_test])
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)
        return doubleloader, testloader
    else:
        return doubleloader

def get_mixedloader_celeba_cartoon(batch_size=100, 
                                   seed=0,
                                   path_to_tgz = '../data/cartoonset10k.tgz', 
                                   extraction_path = '../data',
                                   return_datasets=False,
                                   return_testloader=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    celeba = celeba_dataset()
    cartoon = cartoon_dataset(path_to_tgz, extraction_path)
    print('Datasets loaded')

    cartoon_indices = np.random.choice(len(cartoon), 9000, replace=False)
    cartoon_reduced = torch.utils.data.Subset(cartoon, cartoon_indices)

    celeba_indices = np.random.choice(len(celeba), 9000, replace=False)
    celeba_reduced = torch.utils.data.Subset(celeba, celeba_indices)

    dataset = torch.utils.data.ConcatDataset([cartoon_reduced, celeba_reduced])
    print('Concatenated')
    mixedloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    if return_datasets:
        return mixedloader, cartoon_reduced, celeba_reduced
    elif return_testloader:
        celeba_test_indices = np.setdiff1d(np.arange(len(celeba)), celeba_indices)[:1000]
        celeba_test = torch.utils.data.Subset(celeba, celeba_test_indices)
        cartoon_test_indices = np.setdiff1d(np.arange(len(cartoon)), cartoon_indices)[:1000]
        cartoon_test = torch.utils.data.Subset(cartoon, cartoon_test_indices)
        test_dataset = torch.utils.data.ConcatDataset([cartoon_test, celeba_test])
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)
        return mixedloader, testloader
    else:
        return mixedloader

def get_testloader_celeba_cartoon(batch_size=100,
                                    seed=0,
                                    path_to_tgz = '../data/cartoonset10k.tgz', 
                                    extraction_path = '../data'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    celeba = celeba_dataset()
    cartoon = cartoon_dataset(path_to_tgz, extraction_path)
    print('Datasets loaded')

    cartoon_indices = np.random.choice(len(cartoon), 9000, replace=False)
    cartoon_reduced = torch.utils.data.Subset(cartoon, cartoon_indices)

    celeba_indices = np.random.choice(len(celeba), 9000, replace=False)
    celeba_reduced = torch.utils.data.Subset(celeba, celeba_indices)

    celeba_test_indices = np.setdiff1d(np.arange(len(celeba)), celeba_indices)[:1000]
    celeba_test = torch.utils.data.Subset(celeba, celeba_test_indices)
    cartoon_test_indices = np.setdiff1d(np.arange(len(cartoon)), cartoon_indices)[:1000]
    cartoon_test = torch.utils.data.Subset(cartoon, cartoon_test_indices)

    celeba_nolabels = [data[0] for data in celeba_test]
    cartoon_nolabels = [data[0] for data in cartoon_test]

    test_dataset = CombinedDataset(celeba_nolabels, cartoon_nolabels)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True)
    return testloader


def compute_loss_BLC(logits, z_mu, z_sigma, y, beta=1.0):
    '''
    Compute the loss for the latent classifier.
    '''
    log_likelihood = nn.BCEWithLogitsLoss()(logits, y)
    kl = kl_normal(z_mu, z_sigma, torch.zeros_like(z_mu), torch.ones_like(z_sigma))
    loss = (log_likelihood + beta*kl).mean()
    return loss

class model_counter(nn.Module):
    '''
    After initialization, keep count of the forward passes in the model
    '''
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.counter = 0

    def forward(self, t, x):
        self.counter += 1
        return self.model(t, x)