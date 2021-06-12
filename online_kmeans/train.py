import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader
from scipy.spatial import distance
import torch.optim as optim
from online_kmeans.network import EVT
from online_kmeans.losses import EVTLoss
from torch.optim import lr_scheduler
from sklearn import metrics
from online_kmeans.datasets import BlobsDataset, CustomDataset, irisDataset, LibsvmDataset, MNIST_f, CIFAR10_f


def randCent(dataset, k):
    m = dataset.shape[0]
    choice_idx = np.random.choice(m, k, replace=False)
    centroids = dataset[choice_idx]
    return centroids

def kmeans_plus_plus_Cent(dataset, k):

    m = dataset.shape[0]
    first_idx = np.random.choice(m)
    centroids = dataset[first_idx, :].reshape(1, -1)

    for _ in range(0, k - 1):
        dist = distance.cdist(dataset, centroids, metric='euclidean')
        minDist = dist.min(1)
        minDist2 = np.power(minDist, 2)
        weights = minDist2 / np.sum(minDist2)
        choice_idx = np.random.choice(weights.shape[0], p=weights)
        centroids = np.append(centroids,
                              dataset[choice_idx, :].reshape(1, -1),
                              axis=0)

    return centroids

def Euclidean_distance(tensor0, tensor1):
    tensor0 = tensor0.unsqueeze(1).repeat(1, tensor1.size(0), 1)
    tensor1 = tensor1.repeat(tensor0.size(0),1,1)
    dist = (tensor0-tensor1).pow(2).sum(-1).sqrt()
    return dist

def BMM(data, n_blocks):
    if ((data.shape[0] > n_blocks) or (data.shape[0] == n_blocks)) and (n_blocks > 1):
        block_size = data.shape[0]//n_blocks
        dist_split = torch.split(data, split_size_or_sections=block_size, dim=0)
        dist_max = torch.zeros(n_blocks)
        for i in range(n_blocks):
            dist_max[i], _ = torch.max(dist_split[i], 0, keepdim=True)
    elif ((data.shape[0] > n_blocks) or (data.shape[0] == n_blocks)) and n_blocks == 1:
        dist_max, _ = torch.max(data, 0, keepdim=True)
    elif data.shape[0] < n_blocks:
        dist_max = data
    return dist_max

def POT(data, percent):
    data, _ = torch.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    if split_len == 0:
        if data.shape[0]==0:
            print('csc')
        return data[-1:], data[-1]
    else:
        return data[-split_len:], data[-split_len]


def train(centroids, num_centers, seed, extreme_model, dataset, batch_size):
    # seed
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Dataset
    if dataset == 'MNIST_f':
        train_data = MNIST_f('dataset/MNIST_feature.npy')
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)

    elif dataset == 'CIFAR10_f':
        train_data = CIFAR10_f('dataset/CIFAR10_feature.npy')
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)

    elif dataset == 'iris':
        train_data = irisDataset()
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)
    else:
        dataset_path = os.path.join('dataset', dataset)
        train_data = LibsvmDataset(dataset_path)
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    centroids = torch.from_numpy(centroids).float().to(device)
    count = torch.zeros(num_centers).to(device)
    thres = torch.zeros(num_centers).to(device)
    count_thres = torch.zeros(num_centers).to(device)
    data_over_thres = [[] for _ in range(num_centers)]

    # GEV/GPD network
    evt = EVT(num_centers, extreme_model, device)
    evt = evt.to(device)

    # loss and optimizer
    criterion = EVTLoss()
    optimizer = optim.SGD(evt.parameters(), lr=0.0001, momentum=0.9)
    
    maxindex_list = []
    mindist_list = []
    label_list = []

    # training
    for epoch in range(1):  
        evt.train()
        running_loss = 0.0
        for iter_i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            feature, label = data
            feature = feature.to(device)
            label = label.to(device)
            label_list.append(label)

            dist = Euclidean_distance(feature, centroids)
            
            simlarity = 1- evt.cdf(dist)
            maxsim, maxindex = torch.max(simlarity, dim=-1)
            maxindex_list.append(maxindex)
            mindist = dist[range(dist.size(0)), maxindex]
            mindist_list.append(mindist)

            # update centroids
            for idx, x in enumerate(feature):
                index = maxindex[idx]
                count[index] = count[index] + 1
                eta = 1 / count[index]
                centroids[index] = (1 - eta) * centroids[index] + eta * x

            # extract extreme data
            dist = Euclidean_distance(feature, centroids)
            mindist, minindex = torch.min(dist, dim=-1)
            if extreme_model == 'gev':
                dist_max = []
                for j in range(num_centers):
                    dist_j = mindist[minindex == j]
                    dist_j_max = BMM(dist_j, 30).to(device)
                    dist_max.append(dist_j_max)
            elif extreme_model == 'gpd':
                dist_max = []
                for j in range(num_centers):
                    dist_j = mindist[minindex == j]
                    dist_j_max, thres_n = POT(dist_j, 0.2)
                    thres[j] = thres_n
                    dist_max.append(dist_j_max-thres_n)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = evt(dist_max)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()
            

    maxindex_all = torch.cat(maxindex_list)
    mindist_all = torch.cat(mindist_list)
    label_all = torch.cat(label_list)
    
    label_all = label_all.cpu().numpy().astype(np.int)
    maxindex_all =  maxindex_all.cpu().numpy()
    sse = mindist_all.sum().item()
    print('Finished Training')

    return label_all, maxindex_all, sse
