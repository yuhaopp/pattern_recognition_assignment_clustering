import pandas as pd
import numpy as np
import random


class KMeans:
    def __init__(self, c_clusters):
        self.c_clusters = c_clusters
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

    def fit(self, data):
        self.data = data
        self.get_init_centroids()
        dist_matrix = np.sqrt(
            -2 * np.dot(data, self.centroids.T) + np.sum(np.square(self.centroids), axis=1) + np.transpose(
                [np.sum(np.square(data), axis=1)]))
        # calculate each point's new cluster
        self.clustering_results = np.array([dist_matrix.argmin(axis=1)])

        # calculate each cluster's new centroid
        index_df = np.concatenate((self.clustering_results.T, self.data), axis=1)
        index_df = pd.DataFrame(index_df).T
        index_df.rename(columns={index_df.columns[0]: "index"}, inplace=True)
        index_df.sort_values('index', inplace=True)
        self.centroids = index_df.groupby('index').mean()
