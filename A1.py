import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_and_preprocess_data(file_path):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_path)  # Read CSV file
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values with column mean
    return df

def train_models(df, feature_col, target_cols):
    """Trains linear regression models for multiple target variables."""
    X_train = df[[feature_col]]  # Independent variable
    models = {}
    predictions = {}

    for target in target_cols:
        y_train = df[target]  # Dependent variable
        model = LinearRegression().fit(X_train, y_train)  # Train model
        models[target] = model  # Store trained model
        predictions[target] = model.predict(X_train)  # Store predictions

    return models, predictions

def plot_results(df, feature_col, target_cols, predictions):
    """Plots actual vs predicted values for each blood type."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Linear Regression: Actual vs Predicted Blood Type Percentages", fontsize=14)
    axes = axes.flatten()

    for i, target in enumerate(target_cols):
        ax = axes[i]
        ax.scatter(df[feature_col], df[target], color="blue", label="Actual", alpha=0.5)
        ax.scatter(df[feature_col], predictions[target], color="red", label="Predicted", alpha=0.5)
        ax.set_title(f"{target} Blood Type")
        ax.set_xlabel("Population")
        ax.set_ylabel(f"{target} Percentage")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Define feature and target columns
feature_col = "Population"
target_cols = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

# Train models and get predictions
models, predictions = train_models(df, feature_col, target_cols)

# Plot actual vs predicted values
plot_results(df, feature_col, target_cols, predictions)
