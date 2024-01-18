import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def predict_daily_demand(df):
    
    features = ['product_sold', 'price', 'rating', 'total_rating',  'status']

    # Create a copy of the original DataFrame for prediction
    df_copy = df.copy()

    # Scale the features in the copy using Min-Max scaling
    scaler = MinMaxScaler()
    df_copy[features] = scaler.fit_transform(df_copy[features])

    # Select features and target variable
    X = df_copy[features]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df_copy['customer_value'], test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the entire dataset
    df_copy['predicted_demand'] = model.predict(X)

    # Add noise to each prediction separately
    noise_scale = 0.5  # Set to the desired noise scale
    noise = np.random.normal(loc=0, scale=noise_scale, size=len(df_copy))
    df_copy['predicted_demand'] += np.abs(noise)  

    # Save predicted demand values back to the original DataFrame
    df['predicted_demand'] = df_copy['predicted_demand']

    # Calculate MSE for the test set (you can use the original 'customer_value' for comparison)
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    print(f"Demand MSE {mse_test}")
    # Feature importance
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)

    # print(df.to_string(index=False))
    # Calculate average prediction per day
    avg_predictions_per_day = df.groupby('day')['predicted_demand'].mean().reset_index()
    print(f"daily demand: {avg_predictions_per_day} type: {type(avg_predictions_per_day)}")
    return df, avg_predictions_per_day

    

def calculate_average_customers(df):
        # Total customers per day
        total_customers_per_day = df.groupby('day')['customer_value'].sum().reset_index()

        # Average customers per day
        average_customers_per_day = total_customers_per_day['customer_value'].mean()

        # Mean of each day
        mean_customer_value_per_day = df.groupby('day')['customer_value'].mean().reset_index()

        # print("Total Customers per Day:")
        # print(total_customers_per_day)
        # print("\nAverage Customers per Day:", average_customers_per_day)
        # print("\nMean Customer Value for Each Day:")
        # print(mean_customer_value_per_day)