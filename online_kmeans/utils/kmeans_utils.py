import numpy as np
from scipy.spatial import distance

def distEclud(centroids, data_i):
    dist = np.sqrt(np.sum(np.power(centroids - data_i, 2), 1))
    return dist


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


def kmeans_plus_plus_Center(dataset, k):
    # centroids=[]
    total = 0
    # 首先随机选一个中心点
    firstCenter = np.random.choice(range(dataset.shape[0]))
    centroids = dataset[firstCenter, :]
    centroids = centroids.reshape(1, -1)
    # 选择其它中心点，对于每个点找出离它最近的那个中心点的距离
    for i in range(0, k - 1):
        weights = [
            np.min(distEclud(centroids, dataset[i]))
            for i in range(dataset.shape[0])
        ]
        total = sum(weights)
        # 归一化0到1之间
        weights = [x / total for x in weights]

        num = np.random.random()
        total = 0
        x = -1
        while total < num:
            x += 1
            total += weights[x]
        a = dataset[x, :].reshape(1, -1)
        centroids = np.append(centroids, dataset[x, :].reshape(1, -1), axis=0)
    # self.centroids=[[self.data[i][r] for i in range(1,self.cols)] for r in centroids]
    return np.array(centroids)


def BMM(data, n_blocks):
    if (data.shape[0] > n_blocks) and (n_blocks > 1):
        split_posi = data.shape[0] // n_blocks * (n_blocks - 1)
        last_block_max = np.max(data[split_posi:], 0)

        pre_block = np.split(data[:split_posi], n_blocks - 1)
        pre_block_max = np.max(pre_block, 1)

        dist_block_max = np.vstack((pre_block_max, last_block_max))
    else:
        dist_block_max = np.max(data, 0).reshape(1, -1)

    return dist_block_max


def POT_max(data, percent):
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    return data[-split_len:], data[-split_len]


def POT_min(data, percent):
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    return data[:split_len], data[split_len]
