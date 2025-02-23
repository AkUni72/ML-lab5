import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("bloodtypes.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X_cluster = df[['Population']]

k_values = range(2, 20)
distortions = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300).fit(X_cluster)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Distortion)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()
