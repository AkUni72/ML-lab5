import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_and_preprocess_data(file_path):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_path)  # Read CSV file
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values with column mean
    return df

def perform_kmeans_clustering(df, feature_col, n_clusters=2):
    """Performs K-Means clustering and assigns cluster labels."""
    X_cluster = df[[feature_col]]  # Select feature for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X_cluster)  # Train K-Means model
    
    df['Cluster'] = kmeans.labels_  # Assign cluster labels to dataset
    return df, kmeans.cluster_centers_

def plot_clusters(df, feature_col, cluster_centers):
    """Plots clustered data and cluster centers."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_col], df['Cluster'], c=df['Cluster'], cmap='coolwarm', alpha=0.6, label='Countries')  # Clustered points
    
    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], np.arange(len(cluster_centers)), color='black', marker='*', s=200, label='Cluster Centers')  
    plt.xlabel("Population")
    plt.ylabel("Cluster")
    plt.title("K-Means Clustering of Countries Based on Population")
    plt.legend()
    plt.grid(True)
    plt.show()

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Perform K-Means clustering
df, cluster_centers = perform_kmeans_clustering(df, "Population")

# Plot the clustered data
plot_clusters(df, "Population", cluster_centers)

# Display first few rows with assigned clusters
print(df[['Country', 'Population', 'Cluster']].head())
