import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset

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
        MNISTImagesOnly('data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTImagesOnly('data', train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True
    )
    return train_loader, test_loader

def get_svhn_data():
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='extra', download=True, transform=preprocess),
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
        datasets.SVHN('data', split='extra', download=True, transform=preprocess),
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
        
    d1 = datasets.MNIST('data', train=True, download=True, transform=preprocess)
    d2 = datasets.SVHN('data', split='extra', download=True, transform=preprocess)

    d2_reduced = torch.utils.data.Subset(d2, np.random.choice(len(d2), len(d1), replace=False))

    dataset = torch.utils.data.ConcatDataset([d1, d2_reduced])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    
    return train_loader

def get_doubleloader_mnist_and_svhn_data(batch_size=100):
    preprocess = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.], [1.])
            ])
        
    d1 = datasets.MNIST('data', train=True, download=True, transform=preprocess)
    d2 = datasets.SVHN('data', split='extra', download=True, transform=preprocess)

    d2_reduced = torch.utils.data.Subset(d2, np.random.choice(len(d2), len(d1), replace=False))

    d1_nolabels = [data[0] for data in d1]
    d2_nolabels = [data[0] for data in d2_reduced]

    dataset = CombinedDataset(d1_nolabels, d2_nolabels)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    
    return train_loader




def get_mnist_data(device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
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