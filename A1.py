import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("bloodtypes.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X_train = df[['Population']]
blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

models = {}
predictions = {}

for blood in blood_types:
    y_train = df[blood]
    model = LinearRegression().fit(X_train, y_train)
    models[blood] = model
    predictions[blood] = model.predict(X_train)

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle("Linear Regression: Actual vs Predicted Blood Type Percentages", fontsize=14)
axes = axes.flatten()

for i, blood in enumerate(blood_types):
    ax = axes[i]
    ax.scatter(df["Population"], df[blood], color="blue", label="Actual", alpha=0.5)
    ax.scatter(df["Population"], predictions[blood], color="red", label="Predicted", alpha=0.5)
    ax.set_title(f"{blood} Blood Type")
    ax.set_xlabel("Population")
    ax.set_ylabel(f"{blood} Percentage")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
