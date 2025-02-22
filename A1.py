import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("bloodtypes.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

# Selecting the feature (Population) and targets (Blood types)
X_train = df[['Population']]  # Independent variable
blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

# Dictionary to store models and predictions
models = {}
predictions = {}

# Train Linear Regression for each blood type
for blood in blood_types:
    y_train = df[blood]  # Target variable
    
    # Train the model
    model = LinearRegression().fit(X_train, y_train)
    
    # Store model and predictions
    models[blood] = model
    predictions[blood] = model.predict(X_train)

# Plot actual vs predicted values for each blood type
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle("Linear Regression: Actual vs Predicted Blood Type Percentages", fontsize=14)

# Flatten the axes array for easier indexing
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