import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv("kmeans.csv")

#extract features and normalize
X = df[['x1', 'x2']].values
X_mean, X_std = np.mean(X, 0), np.std(X, 0)
X = (X - X_mean) / X_std  

def dist(a, b):
     #euclidean dist.
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(X, k, iters=100, init_centroids=None):
  
    centers = np.array(init_centroids)

    for _ in range(iters):
        labels = np.array([np.argmin([dist(p, c) for c in centers]) for p in X])
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers

def plot_clusters(X, labels, centers, k):
    #visualize clusters
    plt.figure(figsize=(8, 6))
    
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i+1}")
    
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label="Centroids", s=200)
    
    plt.xlabel("Feature 1 (Norm)")
    plt.ylabel("Feature 2 (Norm)")
    plt.title(f"K-Means Clustering (k={k})")
    plt.legend()
    plt.show()

#initial centroids for k=2 and k=3
init_centroids_2 = [X[0], X[1]]  #first two points as initial centroids
init_centroids_3 = [X[0], X[1], X[2]]  #first three points as initial centroids

#run K-Means for k=2 and k=3 with in initial centroids
labels_2, centers_2 = kmeans(X, k=2, init_centroids=init_centroids_2)
plot_clusters(X, labels_2, centers_2, k=2)

labels_3, centers_3 = kmeans(X, k=3, init_centroids=init_centroids_3)
plot_clusters(X, labels_3, centers_3, k=3)
