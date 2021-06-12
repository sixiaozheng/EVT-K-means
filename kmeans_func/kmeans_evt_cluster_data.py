import numpy as np
from copy import deepcopy
from scipy.spatial import distance
from scipy.stats import genextreme, genpareto, weibull_max


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
        # 归一�?�?之间
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
    if data.shape[0]==0:
        print("sdaf")
    data = np.sort(data, 0)
    data_len = data.shape[0]
    split_len = int(data_len * percent)
    return data[:split_len], data[split_len]


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
    while clusterChanged:
        # compute distances
        dist = distance.cdist(dataset, centroids, metric=distmeas)
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        # if extreme_model == 'gev':
        #     # fit gev
        #     if loc_param:
        #         for j in range(k):
        #             # extract extreme data
        #             dist_j = minDist[minIndex == j].reshape(-1, 1)
        #             if dist_j.shape[0] == 0:
        #                 continue
        #             dist_over_thre, thre = POT_max(-dist_j, POT_k)
        #             c, loc, scale = genextreme.fit(dist_over_thre)
        #             evt_param[j, :] = c, loc, scale
        #     else:
        #         for j in range(k):
        #             # extract extreme data
        #             dist_j = minDist[minIndex == j].reshape(-1, 1)
        #             if dist_j.shape[0] == 0:
        #                 continue
        #             dist_over_thre, thre = POT_max(-dist_j, POT_k)
        #             c, loc, scale = genextreme.fit(dist_over_thre, floc=0)
        #             evt_param[j, :] = c, loc, scale

        #     # compute prob
        #     prob = np.zeros_like(dist)
        #     for j in range(k):
        #         prob[:, j] = genextreme.cdf(-dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

        #     # min prob
        #     minIndex = prob.argmax(1)
        #     minProb = prob.max(1)
        #     # minDist = dist.min(1)
        #     clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'ne_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(-dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 're_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'in_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(dist.shape[-1]):
                prob[:, j] = weibull_max.cdf(1 / dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:, 2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        # update center
        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)

            dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas)
            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob

def k_means_evt_new1(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
    # use the assignment of last iteration
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
    minDist = None
    while clusterChanged:
        if num_iter == 0:
            # compute distances
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            minIndex = dist.argmin(1)
            minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            minDist = dist[range(dist.shape[0]), minIndex]
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            minDist = dist[range(dist.shape[0]), minIndex]
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'ne_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(-dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            minDist = dist[range(dist.shape[0]), minIndex]
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 're_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            minDist = dist[range(dist.shape[0]), minIndex]
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'in_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            dist = distance.cdist(dataset, centroids, metric=distmeas)
            prob = np.zeros_like(dist)
            for j in range(dist.shape[-1]):
                prob[:, j] = weibull_max.cdf(1 / dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            minDist = dist[range(dist.shape[0]), minIndex]
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:, 2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        # update center
        clusterChanged = False
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)

            dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas)
            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob

def k_means_evt_fuzzy(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
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
    while clusterChanged:
        # compute distances
        dist = distance.cdist(dataset, centroids, metric=distmeas)
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'ne_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(-dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 're_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'in_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(dist.shape[-1]):
                prob[:, j] = weibull_max.cdf(1 / dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:, 2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        # update center
        clusterChanged = False
        prob_true = 1 - prob 
        prob_true = prob_true / np.sum(prob_true)
        for j in range(k):
            new_centroid = dataset*prob_true[:,j].reshape(-1,1)
            new_centroid = new_centroid.sum(0)
            # ptsInClust = dataset[clusterAssment[:, 0] == j]
            # new_centroid = np.mean(ptsInClust, axis=0)

            dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas)
            if dist_center > threshold:
                centroids[j, :] = new_centroid
                centroids_hist[j].append(new_centroid)
                clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob

def k_means_evt_new(dataset, k, centroids_init, max_iter=300, distmeas='euclidean', threshold=1e-3, random_state=0, extreme_model='gev', loc_param=False, n_blocks=20, POT_k=0.1):
    # Iterate until the assignment no longer changes
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
    while clusterChanged:
        # compute distances
        dist = distance.cdist(dataset, centroids, metric=distmeas)
        minIndex = dist.argmin(1)
        minDist = dist.min(1)

        if extreme_model == 'gev':
            # fit gev
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_block_max = BMM(dist_j, n_blocks)
                    c, loc, scale = genextreme.fit(dist_block_max, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genextreme.cdf(dist[:, j], evt_param[j, 0], evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'gpd':
            # fit gpd
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre)
                    evt_param[j, :] = c, loc, scale, thre
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    if dist_j.shape[0] == 0:
                        continue
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = genpareto.fit(dist_over_thre-thre, floc=0)
                    evt_param[j, :] = c, loc, scale, thre

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = genpareto.cdf(dist[:, j]-evt_param[j,3], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'ne_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(-dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(-dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 're_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(k):
                prob[:, j] = weibull_max.cdf(dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmin(1)
            minProb = prob.min(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:,2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        elif extreme_model == 'in_re_weibull':
            # fit re_weibull
            if loc_param:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre)
                    evt_param[j, :] = c, loc, scale
            else:
                for j in range(k):
                    # extract extreme data
                    dist_j = minDist[minIndex == j].reshape(-1, 1)
                    dist_over_thre, thre = POT_max(1 / dist_j, POT_k)
                    c, loc, scale = weibull_max.fit(dist_over_thre, floc=0)
                    evt_param[j, :] = c, loc, scale

            # compute prob
            prob = np.zeros_like(dist)
            for j in range(dist.shape[-1]):
                prob[:, j] = weibull_max.cdf(1 / dist[:, j], evt_param[j, 0],evt_param[j, 1], evt_param[j, 2])

            # min prob
            minIndex = prob.argmax(1)
            minProb = prob.max(1)
            # minDist = dist.min(1)
            clusterAssment[:,0], clusterAssment[:,1], clusterAssment[:, 2], clusterAssment[:,3] = minIndex, minProb, minDist, minDist**2

        # update center
        clusterChanged = False
        if (clusterAssment[:, 0] != minIndex).any():    
            clusterChanged = True
        for j in range(k):
            ptsInClust = dataset[clusterAssment[:, 0] == j]
            new_centroid = np.mean(ptsInClust, axis=0)
            centroids[j, :] = new_centroid
            centroids_hist[j].append(new_centroid)

            # dist_center = distance.cdist(new_centroid.reshape(1, -1),centroids[j, :].reshape(1, -1),metric=distmeas)
            # if dist_center > threshold:
            #     centroids[j, :] = new_centroid
            #     centroids_hist[j].append(new_centroid)
            #     clusterChanged = True

        num_iter += 1
        if num_iter >= max_iter:
            print("iteration >= {}".format(max_iter))
            break
    return centroids, clusterAssment, np.array(centroids_hist), num_iter, evt_param, dist, prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    import random
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    random_state = 89  # random.randint(1, 1000)
    n_features = 2
    n_centers = 6
    n_samples = 200 * n_centers
    n_blocks = 30
    POT_k = 0.1
    max_iter = 50

    print(random_state)

    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_centers,
                      random_state=random_state)
    X = StandardScaler().fit_transform(X)

    centroids_random = randCent(X, n_centers)
    centroids_kmeans = kmeans_plus_plus_Cent(X, n_centers)
    # centroids_kmeans1 = kmeans_plus_plus_Center(X, n_centers)

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
    MSE = kmeans.inertia_ / X.shape[0]
    n_iter = kmeans.n_iter_
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(n_iter))

    plt.subplot(261)
    plt.title('sklearn KMeans random')
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
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(262)
    plt.title('gev random')
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

    # plot dist vs. prob
    plt.figure(num=2, figsize=(24, 20))
    n_ch_dist = 30
    m = dist.shape[0]
    choice_idx = np.random.choice(m, n_ch_dist, replace=False)
    dist_choice = dist[choice_idx]
    prob_choice = prob[choice_idx]
    for i in range(n_ch_dist):
        plt.subplot(5, 6, i + 1)
        plt.scatter(dist_choice[i, :], prob_choice[i, :], c='b', s=8)
    plt.figure(1)

    # python k_means_evt gpd random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(263)
    plt.title('gpd random')
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

    # python k_means_evt ne_re_weibull random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt_new(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(264)
    plt.title('gev random fuzzy')
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

    # python k_means_evt re_weibull random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt_new(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(265)
    plt.title('gpd random fuzzy')
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

    # python k_means_evt inverse re_weibull random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
        X,
        n_centers,
        centroids_random,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='in_re_weibull',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(266)
    plt.title('inverse re_weibull random')
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

    # sklearn kmeans++
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
    MSE = kmeans.inertia_ / X.shape[0]
    n_iter = kmeans.n_iter_
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(n_iter))

    plt.subplot(267)
    plt.title('kmeans kmeans++')
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=8)
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='+',
                s=100,
                linewidths=6,
                color='k',
                zorder=10)

    # python k_means_evt gev kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(268)
    plt.title('gev kmeans++')
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

    # python k_means_evt gpd kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
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
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(269)
    plt.title('gpd kmeans++')
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

    # python k_means_evt ne_re_weibull kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='ne_re_weibull',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(2, 6, 10)
    plt.title('ne_re_weibull kmeans++')
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

    # # python k_means_evt re_weibull kmeans++ init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='re_weibull',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(2, 6, 11)
    plt.title('re_weibull kmeans++')
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

    # python k_means_evt inverse re_weibull random init
    start_t = time()
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
        X,
        n_centers,
        centroids_kmeans,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='in_re_weibull',
        loc_param=False,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t
    MSE = np.sum(clusterAssment[:, -1]) / clusterAssment.shape[0]
    print("time: {}s".format(time() - start_t))
    print("MSE: {}".format(MSE))
    print("iter: {}".format(num_iter))

    plt.subplot(2, 6, 12)
    plt.title('inverse re_weibull random')
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

    plt.figure(1)
    plt.savefig('../results/cluster_results_no_loc_10.jpg')  # , dpi=1500, bbox_inches='tight'
    plt.figure(2)
    plt.savefig('../results/prob_vs_dist_no_loc_10.jpg')  # , dpi=1500, bbox_inches='tight'
    plt.show()
