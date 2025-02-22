import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load the dataset
df = pd.read_csv("bloodtypes.csv")  # Update the path if needed

# 2. Fill missing values with column means
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3. Select the feature for clustering (Population)
X_cluster = df[['Population']]

# 4. Apply K-Means Clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_cluster)

# 5. Assign cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# 6. Compute clustering evaluation metrics
silhouette = silhouette_score(X_cluster, kmeans.labels_)
ch_score = calinski_harabasz_score(X_cluster, kmeans.labels_)
db_index = davies_bouldin_score(X_cluster, kmeans.labels_)

# 7. Print evaluation results
print("Clustering Evaluation Metrics:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")