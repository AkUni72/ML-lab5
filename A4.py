import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("bloodtypes.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X_cluster = df[['Population']]

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_cluster)

df['Cluster'] = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(df['Population'], df['Cluster'], c=df['Cluster'], cmap='coolwarm', alpha=0.6, label='Countries')

plt.scatter(cluster_centers[:, 0], [0, 1], color='black', marker='*', s=200, label='Cluster Centers')

plt.xlabel("Population")
plt.ylabel("Cluster")
plt.title("K-Means Clustering of Countries Based on Population")
plt.legend()
plt.grid(True)
plt.show()

print(df[['Country', 'Population', 'Cluster']].head())
