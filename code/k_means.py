import pandas as pd
import numpy as np
import random


class KMeans:
    def __init__(self, c_clusters):
        self.c_clusters = c_clusters
        self.centroids = list
        self.data = list

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
