import pandas as pd
import joblib  # For loading saved models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

cities = ['berlin', 'beijing', 'moscow', 'los_angeles',
          'new_york', 'sydney', 'tokyo', 'london',
          'buenos_Aires', 'mexico']

# Load each city's model
models = {}

for city in cities:
    # Load the entire pipeline (PolynomialFeatures + SVR)
    model = joblib.load(f'models/svr_polynomial_regression/{city}_svr_polynomial_model.pkl')
    models[city] = model

# Create an empty dataframe to hold predictions
df_predictions = pd.DataFrame()

# Load the original weather data for all cities (use the same target variable, MAR, for all)
for city in cities:
    # Handle Mexico and London with special data handling if necessary
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

    # Predict the temperatures for this city using the respective model (pipeline)
    city_predictions = models[city].predict(X)

    # Ensure that the length of city_predictions matches the length of df
    if len(city_predictions) == len(df):
        # Store the predictions in the dataframe
        df_predictions[city] = city_predictions
    else:
        print(f"Skipping {city} due to length mismatch.")

# The target variable (temperature) remains the 'MAR' column from the dataset
target = y

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor()
gb_model.fit(df_predictions, target)

# Evaluate models with Mean Squared Error
gb_predictions = gb_model.predict(df_predictions)

gb_mse = mean_squared_error(target, gb_predictions)

print(f"Gradient Boosting MSE: {gb_mse}")

# Save the models
joblib.dump(gb_model, 'models/combines/gradient_boosting_model_svr_pol_reg.pkl')

print("Ensemble models trained and saved successfully!")
