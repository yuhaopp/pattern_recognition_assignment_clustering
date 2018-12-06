import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics


class KMeans:
    def __init__(self):
        self.c_clusters = 0
        self.centroids = list
        self.data = list
        self.clustering_results = list

    def get_init_centroids(self):
        if self.c_clusters < 1:
            print('please set the number of clusters bigger than 0')
            exit()
        elif self.c_clusters > self.data.__len__():
            print('the number of clusters is not allowed to be set bigger than number of data')
        else:
            self.centroids = np.array(random.sample(self.data.tolist(), self.c_clusters))

    def fit(self, data, c_clusters):
        self.data = data
        self.c_clusters = c_clusters
        self.get_init_centroids()

        convergence = 0
        while convergence != 1:
            # calculate each point's new cluster
            dist_matrix = np.sqrt(
                -2 * np.dot(data, self.centroids.T) + np.sum(np.square(self.centroids), axis=1) + np.transpose(
                    [np.sum(np.square(data), axis=1)]))
            self.clustering_results = np.array([dist_matrix.argmin(axis=1)])

            # calculate each cluster's new centroid
            index_df = np.concatenate((self.clustering_results.T, self.data), axis=1)
            index_df = pd.DataFrame(index_df)
            index_df.rename(columns={index_df.columns[0]: "index"}, inplace=True)
            index_df.sort_values('index', inplace=True)
            new_centroids = index_df.groupby('index').mean().values
            convergence = 1 if (new_centroids == self.centroids).all() else 0
            self.centroids = new_centroids

    def draw_result(self):
        pca = PCA(n_components=2)
        data = pca.fit_transform(self.data)
        plt.scatter(data[:, 0], data[:, 1], c=self.clustering_results[0])
        plt.title('clustering result')
        plt.savefig('clustering result')


if __name__ == '__main__':
    k_means = KMeans()
    # centers = [[2, 2], [1.3, 1.1], [0, 0], [-1.1, 0.9], [0.9, -1.1]]
    # data, labels_true = make_blobs(n_samples=10000, centers=centers, cluster_std=1.2, random_state=0)
    raw_data = pd.read_csv('../data/mobile_train.csv')
    data = raw_data.iloc[:, 1:].values
    labels_true = raw_data.iloc[:, -1:].values[:, 0]
    k_means.fit(data, 4)
    pca = PCA(n_components=2)
    data = pca.fit_transform(k_means.data)
    plt.scatter(data[:, 0], data[:, 1], c=k_means.clustering_results[0])
    plt.title('Clustering result: mobile prices')
    plt.savefig('mobile prices')
    plt.clf()

    plt.scatter(data[:, 0], data[:, 1], c=labels_true)
    plt.title('True classifications: mobile prices')
    plt.savefig('true result mobile prices')

    score = metrics.adjusted_rand_score(labels_true, k_means.clustering_results[0])
    print(score)
