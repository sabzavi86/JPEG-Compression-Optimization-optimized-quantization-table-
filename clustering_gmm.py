# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:04:15 2024

Author: ASUS
Description: Clustering evaluation using different algorithms with PCA for feature reduction.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, LayerNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.image import psnr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture

# Custom PSNR loss function
def psnr_loss(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=255.0)

# Load dataset and model
dataset_ORG = np.load('dataset_ORG.npy', mmap_mode='r')
best_model = tf.keras.models.load_model('best_model.h5', custom_objects={'psnr_loss': psnr_loss})

# Extract features using the dense layer
best_model.summary()
dense_layer_model = Model(inputs=best_model.input, outputs=best_model.get_layer('dense').output)
features = dense_layer_model.predict(dataset_ORG)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Determine the number of PCA components to retain 95% of variance
pca = PCA().fit(normalized_features)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components_95_variance = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {components_95_variance}")

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(cumulative_variance)) + 1, cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.grid()
plt.show()

# Apply PCA with 212 components (as determined in the prior step)
pca = PCA(n_components=212)
reduced_features = pca.fit_transform(normalized_features)

# Range of clusters to evaluate
cluster_range = range(8, 17)

# Initialize dictionaries to store the scores
scores = {
    'KMeans': {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []},
    'GMM': {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []},
    'Agglomerative': {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []},
    'Birch': {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}
}

# Function to evaluate clustering
def evaluate_clustering(labels, features):
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    calinski_harabasz = calinski_harabasz_score(features, labels)
    return silhouette, davies_bouldin, calinski_harabasz

# Perform clustering and evaluation
for n_clusters in cluster_range:
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(reduced_features)
    kmeans_scores = evaluate_clustering(kmeans_labels, reduced_features)
    scores['KMeans']['Silhouette'].append(kmeans_scores[0])
    scores['KMeans']['Davies-Bouldin'].append(kmeans_scores[1])
    scores['KMeans']['Calinski-Harabasz'].append(kmeans_scores[2])
    
    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(reduced_features)
    gmm_scores = evaluate_clustering(gmm_labels, reduced_features)
    scores['GMM']['Silhouette'].append(gmm_scores[0])
    scores['GMM']['Davies-Bouldin'].append(gmm_scores[1])
    scores['GMM']['Calinski-Harabasz'].append(gmm_scores[2])
    
    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(reduced_features)
    agg_scores = evaluate_clustering(agg_labels, reduced_features)
    scores['Agglomerative']['Silhouette'].append(agg_scores[0])
    scores['Agglomerative']['Davies-Bouldin'].append(agg_scores[1])
    scores['Agglomerative']['Calinski-Harabasz'].append(agg_scores[2])
    
    # Birch Clustering
    birch = Birch(n_clusters=n_clusters)
    birch_labels = birch.fit_predict(reduced_features)
    birch_scores = evaluate_clustering(birch_labels, reduced_features)
    scores['Birch']['Silhouette'].append(birch_scores[0])
    scores['Birch']['Davies-Bouldin'].append(birch_scores[1])
    scores['Birch']['Calinski-Harabasz'].append(birch_scores[2])

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for method, method_scores in scores.items():
    axs[0].plot(cluster_range, method_scores['Silhouette'], marker='o', label=method)
    axs[1].plot(cluster_range, method_scores['Davies-Bouldin'], marker='o', label=method)
    axs[2].plot(cluster_range, method_scores['Calinski-Harabasz'], marker='o', label=method)

axs[0].set_title('Silhouette Scores')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('Score')
axs[0].legend()

axs[1].set_title('Davies-Bouldin Scores')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Score')
axs[1].legend()

axs[2].set_title('Calinski-Harabasz Scores')
axs[2].set_xlabel('Number of Clusters')
axs[2].set_ylabel('Score')
axs[2].legend()

plt.tight_layout
