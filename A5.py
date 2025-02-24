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

def perform_kmeans_clustering(df, feature_col, n_clusters=2):
    """Performs K-Means clustering and assigns cluster labels."""
    X_cluster = df[[feature_col]]  # Select feature for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)  # Train K-Means model
    
    df['Cluster'] = kmeans.labels_  # Assign cluster labels to dataset
    return df, kmeans

def evaluate_clustering(X_cluster, labels):
    """Computes clustering evaluation metrics."""
    silhouette = silhouette_score(X_cluster, labels)  # Measure cluster separation
    ch_score = calinski_harabasz_score(X_cluster, labels)  # Measures cluster compactness & separation
    db_index = davies_bouldin_score(X_cluster, labels)  # Lower values indicate better clustering
    
    return silhouette, ch_score, db_index

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Perform K-Means clustering
df, kmeans = perform_kmeans_clustering(df, "Population")

# Evaluate clustering performance
silhouette, ch_score, db_index = evaluate_clustering(df[['Population']], kmeans.labels_)

# Print clustering evaluation metrics
print("Clustering Evaluation Metrics:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
