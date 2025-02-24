import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_and_preprocess_data(file_path):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_path)  # Read CSV file
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values with column mean
    return df

def compute_elbow_method(df, feature_col, k_range):
    """Computes distortions for different k values using the Elbow Method."""
    X_cluster = df[[feature_col]]  # Select feature for clustering
    distortions = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300).fit(X_cluster)  # Train K-Means model
        distortions.append(kmeans.inertia_)  # Inertia (sum of squared distances to cluster centers)

    return k_range, distortions

def plot_elbow_method(k_range, distortions):
    """Plots the Elbow Method graph to determine the optimal number of clusters."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, distortions, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Distortion)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Define range of k values to test
k_range = range(2, 20)

# Compute distortions for different k values
k_values, distortions = compute_elbow_method(df, "Population", k_range)

# Plot Elbow Method results
plot_elbow_method(k_values, distortions)
