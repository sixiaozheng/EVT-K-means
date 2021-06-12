from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs
from sklearn.datasets import load_svmlight_file
from scipy.stats import genextreme, genpareto

class BlobsDataset(Dataset):
    def __init__(self, n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0),
                 shuffle=True, random_state=None):
        data, targets = datasets.make_blobs(n_samples, n_features, centers, cluster_std, center_box, shuffle, random_state)
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx], self.targets[idx]
        return feature, label

class irisDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        data, targets = datasets.load_iris(return_X_y=True)
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.X = np.load(dataset_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature = self.X[idx]
        return feature

class LibsvmDataset(Dataset):
    def __init__(self, dataset_path):
        X_r, self.y = load_svmlight_file(dataset_path)
        self.X = StandardScaler().fit_transform(X_r.A)
        for idx, y_ in enumerate(np.unique(self.y)):
            self.y[self.y == y_] = idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class gevgpdDataset(Dataset):
    def __init__(self, dataset, size):
        if dataset == 'gev':
            self.X = genextreme.rvs(c=3, loc=0, scale=0.5, size=size)
        elif dataset== 'gpd':
            self.X = genpareto.rvs(c=3, loc=0, scale=0.5, size=size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class MNIST_f(Dataset):
    def __init__(self, dataset_path):
        data_np = np.load(dataset_path)
        data = data_np[:, :-1]
        targets = data_np[:, -1]
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx], self.targets[idx]
        return feature, label

class CIFAR10_f(Dataset):
    def __init__(self, dataset_path):
        data_np = np.load(dataset_path)
        data = data_np[:, :-1]
        targets = data_np[:, -1]
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx], self.targets[idx]
        return feature, label