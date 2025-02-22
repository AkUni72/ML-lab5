import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load the dataset
df = pd.read_csv("bloodtypes.csv")  # Update the path if needed

# 2. Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3. Select feature for clustering (Population)
X_cluster = df[['Population']]

# 4. Try different values of k (from 2 to 10) and evaluate clustering
k_values = range(2, 6)  # Testing k from 2 to 6

# Dictionaries to store metric values
silhouette_scores = []
ch_scores = []
db_indices = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
    
    # Compute clustering metrics
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))
    ch_scores.append(calinski_harabasz_score(X_cluster, kmeans.labels_))
    db_indices.append(davies_bouldin_score(X_cluster, kmeans.labels_))

# 5. Plot the results
plt.figure(figsize=(12, 5))

# Silhouette Score Plot
plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.grid(True)

# Calinski-Harabasz Score Plot
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', linestyle='-', color='g')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Calinski-Harabasz Score")
plt.title("CH Score vs k")
plt.grid(True)

# Davies-Bouldin Index Plot
plt.subplot(1, 3, 3)
plt.plot(k_values, db_indices, marker='o', linestyle='-', color='r')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Davies-Bouldin Index")
plt.title("DB Index vs k")
plt.grid(True)

plt.tight_layout()
plt.show()