import numpy as np
from scipy.spatial import distance

def distEclud(centroids, data_i):
    dist = np.sqrt(np.sum(np.power(centroids - data_i, 2), 1))
    return dist


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


def BMM(data, n_blocks):
    if ((data.shape[0] > n_blocks) or (data.shape[0] == n_blocks)) and (n_blocks > 1):
        split_posi = data.shape[0] // n_blocks * (n_blocks - 1)
        last_block_max = np.max(data[split_posi:], 0)

        pre_block = np.split(data[:split_posi], n_blocks - 1)
        pre_block_max = np.max(pre_block, 1)

        dist_block_max = np.vstack((pre_block_max, last_block_max))
    elif ((data.shape[0] > n_blocks) or (data.shape[0] == n_blocks)) and n_blocks == 1:
        dist_block_max = np.max(data, 0).reshape(1, -1)
    elif data.shape[0] < n_blocks:
        dist_block_max = data
    return dist_block_max


def POT_max(data, percent):
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    if split_len == 0:
        return data[-1:], data[-1]
    else:
        return data[-split_len:], data[-split_len]


def POT_min(data, percent):
    if data.shape[0]==0:
        print("sdaf")
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    return data[:split_len], data[split_len]
