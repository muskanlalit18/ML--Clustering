# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 00:25:54 2020

@author: Muskan Lalit
"""

# Heirarchical clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Finding the appropriate amount of clusters using dendrograms
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting heirarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5,linkage='ward',affinity='euclidean' )
y_hc = hc.fit_predict(X)

# Visualizing the results of heirarchical clustering
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()