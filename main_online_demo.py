import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from online_kmeans.datasets import BlobsDataset, CustomDataset, irisDataset, LibsvmDataset, MNIST_f, CIFAR10_f
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader
from scipy.spatial import distance
import torch.optim as optim
from torch.optim import lr_scheduler
from online_kmeans.utils.metrics import get_accuracy
from sklearn import metrics
from online_kmeans.train import train 

def randCent(dataset, k, random_state):
    np.random.seed(random_state)
    m = dataset.shape[0]
    choice_idx = np.random.choice(m, k, replace=False)
    centroids = dataset[choice_idx]
    return centroids

def kmeans_plus_plus_Cent(dataset, k, random_state):
    np.random.seed(random_state)
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


def main(args):
    # seed
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random_state = args.seed

    # Dataset
    if args.dataset == 'MNIST_f':
        train_data = MNIST_f('dataset/MNIST_feature.npy')
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)

    elif args.dataset == 'CIFAR10_f':
        train_data = CIFAR10_f('dataset/CIFAR10_feature.npy')
        train_loader = DataLoader(dataset=train_data,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)
    

    # init centroids
    if args.init_center == 'kmeans++':
        train_data = train_data.data.numpy().reshape(train_data.data.numpy().shape[0], -1)
        centroids = kmeans_plus_plus_Cent(train_data, args.num_centers,random_state)
    elif args.init_center == 'random':
        train_data = train_data.data.numpy().reshape(train_data.data.numpy().shape[0], -1)
        centroids = randCent(train_data, args.num_centers,random_state)

    # online EV-kmeans
    label, pred_y, sse = train(centroids, args.num_centers, args.seed, args.extreme_model, args.dataset, args.batch_size)

    SSE = sse
    MSE = SSE / train_data.shape[0]
    ACC_socre, _ = get_accuracy(label, pred_y, args.num_centers)
    ARI_socre = metrics.adjusted_rand_score(pred_y, label)
    NMI_score = metrics.normalized_mutual_info_score(pred_y, label)
    
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='online EV kmeans')
    parser.add_argument('--batch_size', default=500, type=int, help='batch size')
    parser.add_argument('--epoch', default=1, type=int, help='epoch size')
    parser.add_argument('--samples_per_class', default=500, type=int, help='epoch size')
    parser.add_argument('--num_centers', default=10, type=int, help='num centers')
    parser.add_argument('--save_dir', default='ckpoints', type=str, help='ckpoint loc')
    parser.add_argument('--result_dir', default='outs', type=str, help='output')
    parser.add_argument('--dataset', default='CIFAR10_f', type=str)
    parser.add_argument('--init_center', default='kmeans++', type=str)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--extreme_model', default='gpd', type=str)

    args = parser.parse_args()
    main(args)                                            