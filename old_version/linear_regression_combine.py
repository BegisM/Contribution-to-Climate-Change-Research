import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

cities = ['berlin', 'beijing', 'moscow', 'los_angeles',
          'new_york', 'sydney', 'tokyo', 'london',
          'buenos_Aires', 'mexico']

# Load each city's model
models = {}
for city in cities:
    model = joblib.load(f'models/linear_regression/{city}_linear_model.pkl')
    models[city] = model

# Create an empty dataframe to hold predictions
df_predictions = pd.DataFrame()

# Create a dictionary to hold target values for each city
target_values = {}

# Load the original weather data for all cities
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

    # Collect target (MAR) values for this city
    target_values[city] = y.tolist()

    # Predict the temperatures for this city using the respective model
    city_predictions = models[city].predict(X)

    # Add the city predictions as a new column in df_predictions
    df_predictions[city] = city_predictions

# Now convert the predictions to a single column of predictions (flatten)
# Also, align target values with each city
flattened_predictions = df_predictions.values.flatten()  # Flatten predictions
flattened_target_values = [value for sublist in target_values.values() for value in sublist]  # Flatten target values

# Convert flattened target values into a pd.Series
target = pd.Series(flattened_target_values)

# Convert flattened predictions into a pd.Series
df_predictions_series = pd.Series(flattened_predictions)

# Ensure predictions and targets are aligned (by their index)
df_predictions_series = df_predictions_series.reset_index(drop=True)
target = target.reset_index(drop=True)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(df_predictions_series.values.reshape(-1, 1), target)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor()
gb_model.fit(df_predictions_series.values.reshape(-1, 1), target)

# Evaluate models with Mean Squared Error
rf_predictions = rf_model.predict(df_predictions_series.values.reshape(-1, 1))
gb_predictions = gb_model.predict(df_predictions_series.values.reshape(-1, 1))

rf_mse = mean_squared_error(target, rf_predictions)
gb_mse = mean_squared_error(target, gb_predictions)

print(f"Random Forest MSE: {rf_mse}")
print(f"Gradient Boosting MSE: {gb_mse}")

# Save the models
joblib.dump(rf_model, 'models/combines/random_forest_model_lr.pkl')
joblib.dump(gb_model, 'models/combines/gradient_boosting_model_lr.pkl')

print("Ensemble models trained and saved successfully!")
