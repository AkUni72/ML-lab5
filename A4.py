import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the dataset
df = pd.read_csv("bloodtypes.csv")  # Update path if needed

# 2. Fill missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3. Select the feature for clustering
X_cluster = df[['Population']]

# 4. Apply K-Means Clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_cluster)

# 5. Assign cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# 6. Retrieve cluster centers
cluster_centers = kmeans.cluster_centers_

# 7. Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(
    df['Population'], 
    df['Cluster'], 
    c=df['Cluster'], 
    cmap='coolwarm', 
    alpha=0.6, 
    label='Countries'
)

# 8. Plot the cluster centers
# Note: Since we have only 1D data (Population), 
# we place cluster centers on the x-axis and 
# choose y-coordinates manually for visualization
plt.scatter(
    cluster_centers[:, 0], 
    [0, 1], 
    color='black', 
    marker='*', 
    s=200, 
    label='Cluster Centers'
)

plt.xlabel("Population")
plt.ylabel("Cluster")
plt.title("K-Means Clustering of Countries Based on Population")
plt.legend()
plt.grid(True)
plt.show()

# Optional: Display the first few rows with cluster assignments
print(df[['Country', 'Population', 'Cluster']].head())