import numpy as np
import random
import pandas as pd

a = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
b = np.array([[2, 3], [2, 5], [4, 3], [2, 6]])

dist_matrix = np.sqrt(
    -2 * np.dot(a, b.T) + np.sum(np.square(b), axis=1) + np.transpose(
        [np.sum(np.square(a), axis=1)]))
# calculate each point's new cluster
clustering_results = np.array([dist_matrix.argmin(axis=1)])

index_df = np.concatenate((a, b), axis=1)
index_df = pd.DataFrame(index_df)
t = index_df.iloc[:, 1:].values
index_df.rename(columns={index_df.columns[0]: 'index'}, inplace=True)
means = index_df.groupby('index').mean().iloc[:, 1:].values

print(means)
