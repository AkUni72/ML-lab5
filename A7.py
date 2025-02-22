import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the dataset
df = pd.read_csv("bloodtypes.csv")  # Update the file path if needed

# 2. Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3. Select feature for clustering (Population)
X_cluster = df[['Population']]

# 4. Try different values of k (from 2 to 10) for the Elbow Method
k_values = range(2, 20)  # Testing k from 2 to 20
distortions = []  # To store inertia values

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300).fit(X_cluster)
    distortions.append(kmeans.inertia_)  # Inertia (distortion measure)

# 5. Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Distortion)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()