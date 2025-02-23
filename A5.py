import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("bloodtypes.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X_cluster = df[['Population']]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_cluster)

df['Cluster'] = kmeans.labels_

silhouette = silhouette_score(X_cluster, kmeans.labels_)
ch_score = calinski_harabasz_score(X_cluster, kmeans.labels_)
db_index = davies_bouldin_score(X_cluster, kmeans.labels_)

print("Clustering Evaluation Metrics:")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
