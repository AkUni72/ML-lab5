import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_and_preprocess_data(file_path):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_path)  # Read CSV file
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values with column mean
    return df

def evaluate_kmeans_clustering(df, feature_col, k_range):
    """Evaluates K-Means clustering for different k values."""
    X_cluster = df[[feature_col]]  # Select feature for clustering
    
    silhouette_scores = []
    ch_scores = []
    db_indices = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)  # Train K-Means model
        silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))  # Silhouette Score
        ch_scores.append(calinski_harabasz_score(X_cluster, kmeans.labels_))  # Calinski-Harabasz Score
        db_indices.append(davies_bouldin_score(X_cluster, kmeans.labels_))  # Davies-Bouldin Index

    return k_range, silhouette_scores, ch_scores, db_indices

def plot_clustering_metrics(k_range, silhouette_scores, ch_scores, db_indices):
    """Plots clustering evaluation metrics vs number of clusters."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(k_range, ch_scores, marker='o', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title("CH Score vs k")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(k_range, db_indices, marker='o', linestyle='-', color='r')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("DB Index vs k")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Define range of k values to test
k_range = range(2, 6)

# Evaluate clustering for different k values
k_values, silhouette_scores, ch_scores, db_indices = evaluate_kmeans_clustering(df, "Population", k_range)

# Plot evaluation metrics
plot_clustering_metrics(k_values, silhouette_scores, ch_scores, db_indices)
