import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_path)  # Read CSV file
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing values with column mean
    return df

def train_and_evaluate_models(df, feature_col, target_cols, test_size=0.2):
    """Trains linear regression models and computes evaluation metrics."""
    X = df[[feature_col]]  # Independent variable
    X_train, X_test, y_train_full, y_test_full = train_test_split(X, df[target_cols], test_size=test_size, random_state=42)
    
    metrics = {}  # Store performance metrics

    for target in target_cols:
        y_train = y_train_full[target]
        y_test = y_test_full[target]

        model = LinearRegression().fit(X_train, y_train)  # Train model
        
        y_train_pred = model.predict(X_train)  # Predict on training set
        y_test_pred = model.predict(X_test)  # Predict on test set

        # Compute evaluation metrics
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Store metrics in dictionary
        metrics[target] = {
            "MSE Train": mse_train,
            "MSE Test": mse_test,
            "RMSE Train": rmse_train,
            "RMSE Test": rmse_test,
            "MAPE Train": mape_train,
            "MAPE Test": mape_test,
            "R2 Train": r2_train,
            "R2 Test": r2_test
        }

    return metrics

# Load and preprocess data
df = load_and_preprocess_data("bloodtypes.csv")

# Define feature and target columns
feature_col = "Population"
target_cols = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

# Train models and compute metrics
metrics = train_and_evaluate_models(df, feature_col, target_cols)

# Convert metrics to DataFrame and display results
metrics_df = pd.DataFrame(metrics).T.round(4)
print(metrics_df)
