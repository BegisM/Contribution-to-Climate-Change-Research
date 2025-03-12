import pandas as pd
import joblib  # For loading saved models
from sklearn.metrics import mean_squared_error
import numpy as np

cities = ['berlin', 'beijing', 'moscow', 'los_angeles', 'new_york', 'sydney', 'tokyo', 'london', 'buenos_Aires', 'mexico']

# Load each city's linear regression model
models = {}
for city in cities:
    model = joblib.load(f'models/linear_regression/{city}_linear_model.pkl')
    models[city] = model

# Create an empty dataframe to hold predictions
df_predictions = pd.DataFrame()

# Load the original weather data for all cities and calculate MSE for each model
mse_scores = {}  # To store the MSE for each city (model performance)
target = None

for city in cities:
    if city == 'london' or city == 'mexico':
        if city == 'london':
            df = pd.read_csv(f'weather/london_1960-.csv')
        else:
            df = pd.read_csv(f'weather/mexico_1921.csv')
        df = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2010)].dropna(subset=['MAR'])
    else:
        df = pd.read_csv(f'weather/{city}.csv')
        df = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2010)].dropna(subset=['MAR'])

    X = df[['YEAR']]
    y = df['MAR']

    # Predict the temperatures for this city using the respective model
    city_predictions = models[city].predict(X)

    # Store the predictions in the dataframe
    df_predictions[city] = city_predictions

    # Save the target variable (MAR) for evaluating the model
    target = y

    # Calculate MSE for this model
    mse = mean_squared_error(y, city_predictions)
    mse_scores[city] = mse

# Calculate the weights based on MSE: Inverse of MSE, normalized
total_mse = sum(1 / mse for mse in mse_scores.values())  # Sum of the inverse MSE
model_weights = {city: (1 / mse_scores[city]) / total_mse for city in cities}  # Normalize the weights

# Calculate the weighted average predictions
weighted_predictions = np.zeros(len(target))

for city in cities:
    weighted_predictions += model_weights[city] * df_predictions[city]

# Evaluate the weighted model
weighted_mse = mean_squared_error(target, weighted_predictions)

print(f"Weighted Averaging Model MSE: {weighted_mse}")

# Save the weighted predictions (this is for reference)
joblib.dump(weighted_predictions, 'models/combines/weighted_averaging_model_lr.pkl')

print("Weighted Averaging model predictions saved successfully!")
