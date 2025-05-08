# -*- coding: utf-8 -*-
"""Simplified KNN and K-Means on Digits Dataset with Minimal Visualization"""

# Import required libraries
import matplotlib.pyplot as plt  # For plotting and visualization
from sklearn.datasets import load_digits  # For loading the digits dataset
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction

# Load dataset
digits = load_digits()  # Loads handwritten digit images (8x8 grayscale pixels)
X = digits.data  # Flattened feature matrix (each image as a vector of 64 pixel values)
y = digits.target  # Labels (0 to 9)

# Train-test split for KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Create KNN with k=3
knn.fit(X_train, y_train)  # Fit model on training data
y_pred = knn.predict(X_test)  # Predict labels for test data

# Evaluation
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# K-Means clustering on full data
kmeans = KMeans(n_clusters=3, random_state=0)  # Cluster into 3 groups
kmeans.fit(X)  # Fit K-Means to all the data
kmeans_pred = kmeans.labels_  # Cluster labels assigned to each sample
print(f'K-Means Inertia: {kmeans.inertia_}')  # Inertia measures compactness of clusters

# Reduce dimensions for visualization
pca = PCA(n_components=2)  # Reduce to 2D for plotting
X_pca = pca.fit_transform(X)  # Transform original data
centers_pca = pca.transform(kmeans.cluster_centers_)  # Transform cluster centers

# Plot only K-Means clustering
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pred, cmap='viridis', edgecolors='k', s=30)  # Data points
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=100)  # Cluster centers
plt.title('K-Means Clustering (PCA Projection)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.tight_layout()
plt.show()
