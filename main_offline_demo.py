import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from time import time
import os
import sys
import pickle
from copy import deepcopy
from kmeans_func.utils.metrics import get_accuracy
from kmeans_func.utils.kmeans_utils import distEclud, randCent, kmeans_plus_plus_Cent, BMM, POT_max, POT_min
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, load_iris
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

from kmeans_func.kmeans_evt_evm import k_means_evt_evm

def main(args):
    plot_fig = args.plot_fig
    dataset = args.dataset
    ####################################################################################
    # Dataset
    if dataset == 'blobs': 
        n_features = 2
        n_centers = 5
        n_samples = 100 * n_centers
        seed = 381
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=seed)
        X = StandardScaler().fit_transform(X)
    elif dataset == 'blobs60000': 
        n_features = 2
        n_centers = 15
        n_samples = 4000 * n_centers
        seed = 381
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=seed)
        X = StandardScaler().fit_transform(X)

    save_dir = os.path.join('results', dataset, 'offline')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # max_iter = 100
    # POT_k = 0.1
    random_state = 286 
    print(random_state)
    np.random.seed(random_state)

    ####################################################################################
    # initial center
    centroids_random = randCent(X, n_centers, random_state)
    centroids_kmeans = kmeans_plus_plus_Cent(X, n_centers, random_state)


    #####################################################################################
    # GEV
    block_size = 30
    n_blocks = int(n_samples / block_size)
    dist_threshold = 0.1
    max_iter = 300
    POT_k = 0.3
    # Kmeans+GEV+BMM+random
    centroids_init = deepcopy(centroids_random)

    start_t = time()
    # centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
    #     X,
    #     n_centers,
    #     centroids_init,
    #     max_iter=max_iter,
    #     random_state=random_state,
    #     extreme_model='gev',
    #     loc_param=False,
    #     n_blocks=n_blocks,
    #     POT_k=POT_k)
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_init,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    if plot_fig:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=169, linewidths=6, color='k', zorder=10)
        plt.title("K-Means+GEV random")
        # plt.title(
        #     "K-Means+GEV with scale param random t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(
        #         t, MSE_gev, num_iter,
        #         mean_fit_time))
        sub_save_dir = os.path.join(save_dir, 'K-Means+GEV_random')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(sub_save_dir, 'K-Means+GEV_random_seed{}_block{:.1f}.png'.format(random_state, block_size)), dpi=200, format='png', bbox_inches='tight')
        plt.cla()

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])

    print('# Kmeans+GEV+BMM+random')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))

    ###################################################################################
    # Kmeans+GEV+BMM+kmeans++
    centroids_init = deepcopy(centroids_kmeans)

    start_t = time()
    # centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
    #     X,
    #     n_centers,
    #     centroids_init,
    #     max_iter=max_iter,
    #     random_state=random_state,
    #     extreme_model='gev',
    #     loc_param=False,
    #     n_blocks=n_blocks,
    #     POT_k=POT_k)
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_init,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gev',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    t = time() - start_t

    if plot_fig:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=169, linewidths=6, color='k', zorder=10)
        plt.title("K-Means+GEV kmeans++")
        # plt.title(
        #     "K-Means+GEV with scale param kmeans++ t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(
        #         end_t - start_t, MSE_gev, num_iter,
        #         mean_fit_time))
        sub_save_dir = os.path.join(save_dir, 'K-Means+GEV_kmeans++')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(sub_save_dir, 'K-Means+GEV_kmeans++_seed{}_block{:.1f}.png'.format(random_state, block_size)), dpi=200, format='png', bbox_inches='tight')
        plt.cla()

    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])

    print('# Kmeans+GEV+BMM+kmeans++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))

    ########################################################################################################################
    # GPD
    block_size = 10  # [5, 30, 150]
    n_blocks = int(n_samples / block_size)
    dist_threshold = 0.1
    max_iter = 300
    POT_k = 0.3
#######################################################################################################################
    # Kmeans+GPD+POT random
    centroids_init = deepcopy(centroids_random)

    start_t = time()
    # try:
    # centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
    #     X,
    #     n_centers,
    #     centroids_init,
    #     max_iter=max_iter,
    #     random_state=random_state,
    #     extreme_model='gpd',
    #     loc_param=False,
    #     n_blocks=n_blocks,
    #     POT_k=POT_k)

    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_init,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)

    # except:
    #     continue
    t = time() - start_t

    if plot_fig:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=169, linewidths=6, color='k', zorder=10)
        plt.title("K-Means+GPD random")
        # plt.title(
        #     "K-Means+GPD random t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t, MSE_gev,
        #                                                                                   num_iter,
        #                                                                                   mean_fit_time))
        sub_save_dir = os.path.join(save_dir, 'K-Means+GPD_random')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(sub_save_dir, 'K-Means+GPD_random_seed{}_POT_k{:.2f}.png'.format(random_state, POT_k)), dpi=200, format='png', bbox_inches='tight')
        plt.cla()


    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])


    print('# Kmeans+GPD+POT random')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))

    ########################################################################################################################
    # Kmeans+GPD+POT kmeans++
    centroids_init = deepcopy(centroids_kmeans)

    start_t = time()
    # try:
    # centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob = k_means_evt(
    #     X,
    #     n_centers,
    #     centroids_init,
    #     max_iter=max_iter,
    #     random_state=random_state,
    #     extreme_model='gpd',
    #     loc_param=False,
    #     n_blocks=n_blocks,
    #     POT_k=POT_k)
    
    centroids, clusterAssment, centroids_hist, num_iter, evt_param, dist, prob, prob_hist, sse_hist, label_hist = k_means_evt_evm(
        X,
        n_centers,
        centroids_init,
        max_iter=max_iter,
        random_state=random_state,
        extreme_model='gpd',
        loc_param=True,
        n_blocks=n_blocks,
        POT_k=POT_k)
    # except:
    #     continue
    end_t = time()

    if plot_fig:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusterAssment[:, 0], s=8)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=169, linewidths=6, color='k', zorder=10)
        plt.title("K-Means+GPD kmeans++")
        # plt.title(
        #     "K-Means+GPD kmeans++ t:{:.2f}s, MSE:{:.2f}, n_iter:{}, m_fit_t:{:.2f}s".format(end_t - start_t,
        #                                                                                     MSE_gev,
        #                                                                                     num_iter,
        #                                                                                     mean_fit_time))
        sub_save_dir = os.path.join(save_dir, 'K-Means+GPD_kmeans++')
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(sub_save_dir, 'K-Means+GPD_kmeans++_seed{}_POT_k{:.2f}.png'.format(random_state, POT_k)), dpi=200, format='png', bbox_inches='tight')
        plt.cla()


    # evaluation
    SSE = np.sum(clusterAssment[:, -1])
    MSE = SSE / clusterAssment.shape[0]
    ACC_socre, _ = get_accuracy(clusterAssment[:, 0].astype(np.int), y, n_centers)
    ARI_socre = metrics.adjusted_rand_score(y, clusterAssment[:, 0].astype(np.int))
    NMI_score = metrics.normalized_mutual_info_score(y, clusterAssment[:, 0])

    print('# Kmeans+GPD+POT kmeans++')
    print("SSE:{}".format(SSE))
    print("MSE:{}".format(MSE))
    print("ACC_socre:{}".format(ACC_socre))
    print("ARI_socre:{}".format(ARI_socre))
    print("NMI_socre:{}".format(NMI_score))

    
    print(random_state)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EV k-means')
    parser.add_argument('--dataset', default='blobs', type=str, help='dataset')
    parser.add_argument('--plot_fig', dest='plot_fig', action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)