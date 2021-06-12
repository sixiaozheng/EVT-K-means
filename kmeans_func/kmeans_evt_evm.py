import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from scipy.spatial import distance
from scipy.stats import genextreme, genpareto, weibull_max, weibull_min, norm
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from sklearn import metrics
# from kmeans_func.utils.metrics import get_accuracy
from .utils.metrics import get_accuracy
from sklearn.datasets import make_blobs, load_iris
from sklearn.datasets import load_svmlight_file

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
    firstCenter = np.random.choice(range(dataset.shape[0]))
    centroids = dataset[firstCenter, :]
    centroids = centroids.reshape(1, -1)
    for i in range(0, k - 1):
        weights = [
            np.min(distEclud(centroids, dataset[i]))
            for i in range(dataset.shape[0])
        ]
        total = sum(weights)

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
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    if split_len == 0:
        return data[0], data[0]
    else:
        return data[:split_len], data[split_len]


def k_means_evt_hist(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
    # only use the centroids of last iteration 
    # label_hist is used to record the results after each iteration 
    # to facilitate the study of the relationship between performance (ACC) and the number of iterations.
    np.random.seed(random_state)
    m = dataset.shape[0]
    clusterAssment = np.zeros((m, 4))
    centroids = deepcopy(centroids_init)

    # record centroids (centroids history)
    centroids_hist = [[] for _ in range(k)]
    centroids_cp = deepcopy(centroids_init)
    for j in range(k):
        centroids_hist[j].append(centroids_cp[j, :])

    # EVT param
    if extreme_model == 'gpd':
        evt_param = np.zeros((k, 4))
    else:
        evt_param = np.zeros((k, 3))

    clusterChanged = True
    num_iter = 0
    prob_hist = []
    sse_hist = []
    label_hist = []
    while clusterChanged:
        # compute distances
        if distmeas == 'mahalanobis':
            cov = np.cov(dataset.T)
            inv_cov=inv(cov)
            dist = distance.cdist(dataset, centroids, metric=distmeas, VI=inv_cov)
        elif distmeas == 'euclidean^2':
            dist = distance.cdist(dataset, centroids, metric='euclidean')**2
        else:
            dist = distance.cdist(dataset, centroids, metric=distmeas) # cosine, euclidean
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = 1-genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # max prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))
            label_hist.append(maxIndex)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = 1-genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])
                # prob[:, j] = 1-genpareto.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))
            label_hist.append(maxIndex)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

        # update center
        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)
            
            if distmeas == 'mahalanobis':
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas, VI=inv_cov)
            elif distmeas == 'euclidean^2':
                dist_center = distance.cdist(new_centroid.reshape(1, -1), centroids[j, :].reshape(1, -1), metric='euclidean')**2
            else:
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1), metric=distmeas)

            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist

def k_means_evt(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
    # only use the centroids of last iteration
    np.random.seed(random_state)
    m = dataset.shape[0]
    clusterAssment = np.zeros((m, 4))
    centroids = deepcopy(centroids_init)

    # record centroids (centroids history)
    centroids_hist = [[] for _ in range(k)]
    centroids_cp = deepcopy(centroids_init)
    for j in range(k):
        centroids_hist[j].append(centroids_cp[j, :])

    # EVT param
    if extreme_model == 'gpd':
        evt_param = np.zeros((k, 4))
    else:
        evt_param = np.zeros((k, 3))

    clusterChanged = True
    num_iter = 0
    prob_hist = []
    sse_hist = []
    while clusterChanged:
        # compute distances
        if distmeas == 'mahalanobis':
            cov = np.cov(dataset.T)
            inv_cov=inv(cov)
            dist = distance.cdist(dataset, centroids, metric=distmeas, VI=inv_cov)
        elif distmeas == 'euclidean^2':
            dist = distance.cdist(dataset, centroids, metric='euclidean')**2
        else:
            dist = distance.cdist(dataset, centroids, metric=distmeas) # cosine, euclidean
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = 1-genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # max prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))

            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    # dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_j = dist[minIndex == j,j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = 1-genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])
                # prob[:, j] = 1-genpareto.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))
            
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

        # update center
        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)
            
            if distmeas == 'mahalanobis':
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas, VI=inv_cov)
            elif distmeas == 'euclidean^2':
                dist_center = distance.cdist(new_centroid.reshape(1, -1), centroids[j, :].reshape(1, -1), metric='euclidean')**2
            else:
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1), metric=distmeas)

            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob, prob_hist, sse_hist

def k_means_evt_evm(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
    # only use the centroids of last iteration
    # 1. Class labels are assigned like k-means, 2.parameters of GEV and GPD are fitted, 3.Class labels are assigned according to GEV, GPD 

    np.random.seed(random_state)
    m = dataset.shape[0]
    clusterAssment = np.zeros((m, 4))
    centroids = deepcopy(centroids_init)

    # record centroids (centroids history)
    centroids_hist = [[] for _ in range(k)]
    centroids_cp = deepcopy(centroids_init)
    for j in range(k):
        centroids_hist[j].append(centroids_cp[j, :])

    # EVT param
    if extreme_model == 'gpd':
        evt_param = np.zeros((k, 4))
    else:
        evt_param = np.zeros((k, 3))

    clusterChanged = True
    num_iter = 0
    prob_hist = []
    sse_hist = []
    label_hist = []
    while clusterChanged:
        # compute distances
        if distmeas == 'mahalanobis':
            cov = np.cov(dataset.T)
            inv_cov=inv(cov)
            dist = distance.cdist(dataset, centroids, metric=distmeas, VI=inv_cov)
        elif distmeas == 'euclidean^2':
            dist = distance.cdist(dataset, centroids, metric='euclidean')**2
        else:
            dist = distance.cdist(dataset, centroids, metric=distmeas)
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = dist[minIndex != j,j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(-dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale

            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = dist[minIndex != j,j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(-dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genextreme.cdf(-dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # min prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))
            label_hist.append(maxIndex)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = dist[minIndex != j,j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre

            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = dist[minIndex != j,j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genpareto.cdf(-dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            maxIndex = prob.argmax(1)
            maxProb = prob.max(1)
            prob_hist.append(maxProb.sum())
            sse_hist.append(np.sum(minDist**2))
            label_hist.append(maxIndex)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2



        # update center
        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)

            if distmeas == 'mahalanobis':
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas, VI=inv_cov)
            elif distmeas == 'euclidean^2':
                dist_center = distance.cdist(new_centroid.reshape(1, -1), centroids[j, :].reshape(1, -1), metric='euclidean')**2
            else:
                dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1), metric=distmeas)

            # dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas)

            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    
    dist = distance.cdist(dataset, centroids, metric=distmeas)

    if extreme_model == 'gev':
        for j in range(k):
            # extract extreme data
            dist_j = dist[minIndex != j,j].reshape(-1, 1)
            if dist_j.shape[0] == 0:
                continue
            dist_block_max = BMM(-dist_j, n_blocks)
            # plt.figure(figsize=(4,4))
            # res = stats.probplot(dist_block_max.reshape(-1), dist=genextreme, plot=plt, sparams=(evt_param[j,0],))

            # save_dir = os.path.join('results', 'blobs', 'qqplot')
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_path = os.path.join(save_dir, "qqplot_gev_kmeans{}_{}.png".format(random_state, j))
            # plt.savefig(save_path, dpi=1000, format='png', bbox_inches='tight')
            # plt.show()
    elif extreme_model == 'gpd':
        for j in range(k):
            # extract extreme data
            dist_j = dist[minIndex != j,j].reshape(-1, 1)
            if dist_j.shape[0] == 0:
                continue
            dist_over_thre, thre = POT_max(-dist_j, POT_k)
            # plt.figure(figsize=(4,4))
            # res = stats.probplot(dist_over_thre.reshape(-1), dist=genpareto, plot=plt, sparams=(evt_param[j,0],))

            # save_dir = os.path.join('results', 'blobs', 'qqplot')
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_path = os.path.join(save_dir, "qqplot_gpd_kmeans{}_{}.png".format(random_state, j))
            # plt.savefig(save_path, dpi=1000, format='png', bbox_inches='tight')
            # plt.show()


    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist

def prob_one_zero_in_cluster(dataset, k, centroids, evt_param, extreme_model, distmeas='euclidean'):
    m = dataset.shape[0]
    clusterAssment = np.zeros((m, 4))
    dist = distance.cdist(dataset, centroids, metric=distmeas)
    minIndex = dist.argmin(1)
    minDist = dist.min(1)

    if extreme_model == 'gev':
        # compute prob
        prob = np.zeros_like(dist)
        for j in range(k):
            prob[:, j] = genextreme.cdf(-dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

        # min prob
        maxIndex = prob.argmax(1)
        maxProb = prob.max(1)
        probisOne = maxProb==1.0
        probisZero = maxProb==0.0
        print("{}/{} prob is 1.0".format(sum(maxProb==1.0),maxProb.shape[0]))
        print("{}/{} prob is 0.0".format(sum(maxProb==0.0),maxProb.shape[0]))
        # minDist = dist.min(1)
        clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2

    elif extreme_model == 'gpd':
        # compute prob
        prob = np.zeros_like(dist)
        for j in range(k):
            prob[:, j] = genpareto.cdf(-dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

        # min prob
        maxIndex = prob.argmax(1)
        maxProb = prob.max(1)
        probisOne = maxProb==1.0
        probisZero = maxProb==0.0
        print("{}/{} prob is 1.0".format(sum(maxProb==1.0),maxProb.shape[0]))
        print("{}/{} prob is 0.0".format(sum(maxProb==0.0),maxProb.shape[0]))
        # minDist = dist.min(1)
        clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = maxIndex, maxProb, minDist, minDist**2
    return centroids, clusterAssment, evt_param, probisOne, probisZero


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    import random
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    # random_state = 89 #random.randint(1, 1000) #123#89  # random.randint(1, 1000)
    # n_features = 2
    # n_centers = 3
    # n_samples = 200 * n_centers
    # n_blocks = 30
    # POT_k = 0.2
    # max_iter = 50

    # print(random_state)
    X_r, y = load_svmlight_file('../dataset/sonar_scale')
    X=X_r.A
    for idx, y_ in enumerate(np.unique(y)):
        y[y==y_] = idx

    # X, y = load_iris(return_X_y=True)
    random_state = 402
    n_features = X.shape[-1]
    n_centers = len(np.unique(y))
    n_samples = X.shape[0]
    n_blocks = 30
    POT_k = 0.2
    max_iter = 50

    # X, y = make_blobs(n_samples=n_samples,
    #                   n_features=n_features,
    #                   centers=n_centers,
    #                   random_state=random_state)
    X = StandardScaler().fit_transform(X)

    # def generate_gaussian(n_sample, n_cluster, n_d, sigma, std, seed):
    #     centroids = np.random.uniform(-sigma, sigma, (n_cluster, n_d))
    #     X_l = []
    #     y_l=[]
    #     for i in range(n_cluster):
    #         X_i=centroids[i,:]+np.random.normal(loc=0.0, scale=std, size=(int(n_sample/n_cluster), n_d))
    #         y_i=(i+1)*np.ones(int(n_sample/n_cluster))
    #         X_l.append(X_i)
    #         y_l.append(y_i)
    #     X=np.concatenate(X_l)
    #     y=np.concatenate(y_l)
    #     return X, y
    # X, y = generate_gaussian(1000, 4, 2, 1, 0.3, random_state)
    # n_features = X.shape[1]
    # n_centers = len(np.unique(y))
    # n_samples = len(X)
    # n_blocks = 30
    # POT_k = 0.2
    # max_iter = 50

    centroids_random = randCent(X, n_centers)
    centroids_kmeans = kmeans_plus_plus_Cent(X, n_centers)

    plt.figure(num=1, figsize=(25, 10))

    # sklearn KMeans random init
    start_t = time()
    kmeans = KMeans(n_clusters=n_centers,
                    init=centroids_random,
                    n_init=1,
                    max_iter=max_iter,
                    precompute_distances=False,
                    verbose=10,
                    algorithm='full',
                    random_state=random_state).fit(X)
    t = time() - start_t
    n_iter = kmeans.n_iter_

    SSE = kmeans.inertia_
    MSE = SSE / X.shape[0]
    ACC_socre, _ = get_accuracy(kmeans.labels_, y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, kmeans.labels_)
    NMI_score = metrics.normalized_mutual_info_score(y, kmeans.labels_)
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('sklearn KMeans random')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(451)
    plt.title('sklearn KMeans random')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)

    # sklearn KMeans kmeans++ init
    start_t = time()
    kmeans = KMeans(n_clusters=n_centers,
                    init=centroids_kmeans,
                    n_init=1,
                    max_iter=max_iter,
                    precompute_distances=False,
                    verbose=10,
                    algorithm='full',
                    random_state=random_state).fit(X)
    t = time() - start_t
    n_iter = kmeans.n_iter_

    SSE = kmeans.inertia_
    MSE = SSE / X.shape[0]
    ACC_socre, _ = get_accuracy(kmeans.labels_, y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, kmeans.labels_)
    NMI_score = metrics.normalized_mutual_info_score(y, kmeans.labels_)
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('sklearn KMeans kmeans++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))


    plt.subplot(456)
    plt.title('sklearn KMeans kmeans++')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)

    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gev')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(452)
    plt.title('k_means_evt_evm gev')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param,extreme_model='gev')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(453)
    # plt.title('k_means_evt_evm gev random p 1 0')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gev ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(457)
    plt.title('k_means_evt_evm gev ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param, extreme_model='gev')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(458)
    # plt.title('k_means_evt_evm gev ++ p 1 0')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)

    # plot dist vs. prob
    # plt.figure(num=2, figsize=(24, 20))
    # n_ch_dist = 30
    # m = dist.shape[0]
    # choice_idx = np.random.choice(m, n_ch_dist, replace=False)
    # dist_choice = dist[choice_idx]
    # prob_choice = prob[choice_idx]
    # for i in range(n_ch_dist):
    #     plt.subplot(5, 6, i + 1)
    #     plt.scatter(dist_choice[i, :], prob_choice[i, :], c='b', s=8)
    # plt.figure(1)


    # python k_means_evt gpd random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gpd')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(453)
    plt.title('k_means_evt_evm gpd')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param,extreme_model='gpd')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(455)
    # plt.title('k_means_evt_evm gpd')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)


    # python k_means_evt evm gpd kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gpd ++ ')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,8)
    plt.title('k_means_evt_evm gpd ++ ')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids in centroids_hist:
        centroids = np.vstack(centroids)
        plt.plot(centroids[:, 0], centroids[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param, extreme_model='gpd')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(4,5,10)
    # plt.title('k_means_evt_evm gpd ++')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)

    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gev')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,4)
    plt.title('k_means_evt gev')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param,extreme_model='gev')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(4,5,13)
    # plt.title('k_means_evt_evm gev random p 1 0')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gev ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,9)
    plt.title('k_means_evt gev ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # prob one zero 
    # start_t = time()
    # centroids, clusterAssment, evt_param, probisOne, probisZero = prob_one_zero_in_cluster(X, n_centers, centroids, evt_param, extreme_model='gev')
    # t = time() - start_t
    # MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    # print("time: {}s".format(t))
    # print("MSE: {}".format(MSE))

    # plt.subplot(4,5,18)
    # plt.title('k_means_evt_evm gev ++ p 1 0')
    # plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    # plt.scatter(centroids[:, 0],
    #             centroids[:, 1],
    #             marker='+',
    #             s=100,
    #             linewidths=6,
    #             color='k',
    #             zorder=10)
    # for centroids_ in centroids_hist:
    #     centroids_ = np.vstack(centroids_)
    #     plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # for i in range(X.shape[0]):
    #     if probisOne[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)
    #     if probisZero[i]:
    #         plt.scatter(X[i, 0], X[i, 1], c='k', s=8)

    # plot dist vs. prob
    # plt.figure(num=2, figsize=(24, 20))
    # n_ch_dist = 30
    # m = dist.shape[0]
    # choice_idx = np.random.choice(m, n_ch_dist, replace=False)
    # dist_choice = dist[choice_idx]
    # prob_choice = prob[choice_idx]
    # for i in range(n_ch_dist):
    #     plt.subplot(5, 6, i + 1)
    #     plt.scatter(dist_choice[i, :], prob_choice[i, :], c='b', s=8)
    # plt.figure(1)


    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gpd')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,5)
    plt.title('k_means_evt gpd')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gpd ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,10)
    plt.title('k_means_evt gpd ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


##################################################################################################

    # sklearn KMeans random init
    start_t = time()
    kmeans = KMeans(n_clusters=n_centers,
                    init=centroids_random,
                    n_init=1,
                    max_iter=max_iter,
                    precompute_distances=False,
                    verbose=10,
                    algorithm='full',
                    random_state=random_state).fit(X)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# sklearn KMeans random')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,11)
    plt.title('sklearn KMeans random')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)

    # sklearn KMeans kmeans++ init
    start_t = time()
    kmeans = KMeans(n_clusters=n_centers,
                    init=centroids_kmeans,
                    n_init=1,
                    max_iter=max_iter,
                    precompute_distances=False,
                    verbose=10,
                    algorithm='full',
                    random_state=random_state).fit(X)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# sklearn KMeans kmeans++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,16)
    plt.title('sklearn KMeans kmeans++')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)

    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gev')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,12)
    plt.title('k_means_evt_evm gev')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gev ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,17)
    plt.title('k_means_evt_evm gev ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    # python k_means_evt gpd random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
    
    print('# k_means_evt_evm gpd')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,13)
    plt.title('k_means_evt_evm gpd')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    # python k_means_evt evm gpd kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt_evm gpd ++ ')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,18)
    plt.title('k_means_evt_evm gpd ++ ')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids in centroids_hist:
        centroids = np.vstack(centroids)
        plt.plot(centroids[:, 0], centroids[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gev')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))


    plt.subplot(4,5,14)
    plt.title('k_means_evt gev')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gev ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,19)
    plt.title('k_means_evt gev ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    # python k_means_evt gev random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gpd')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))


    plt.subplot(4,5,15)
    plt.title('k_means_evt gpd')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])
    silhouette_score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')

    print('# k_means_evt gpd ++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))
    print("silhouette_score:{}".format(silhouette_score))

    plt.subplot(4,5,20)
    plt.title('k_means_evt gpd ++')
    plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)
    for centroids_ in centroids_hist:
        centroids_ = np.vstack(centroids_)
        plt.plot(centroids_[:, 0], centroids_[:, 1], 'k-', marker='o', markersize=4, linewidth=2)


    

    plt.figure(1)
    plt.savefig('../results/cluster_results_evmnew_519.jpg')  # , dpi=1500, bbox_inches='tight'
    plt.figure(2)
    # plt.savefig('../results/prob_vs_dist_no_loc_10.jpg')  # , dpi=1500, bbox_inches='tight'
    plt.show()
