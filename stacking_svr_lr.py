import pandas as pd
import joblib  # For loading saved models
from sklearn.svm import SVR  # Importing SVR for stacking
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

cities = ['berlin', 'beijing', 'moscow', 'los_angeles', 'new_york', 'sydney', 'tokyo', 'london', 'buenos_Aires', 'mexico']

# Load each city's linear regression model
models = {}
for city in cities:
    model = joblib.load(f'models/linear_regression/{city}_linear_model.pkl')
    models[city] = model

# Create an empty dataframe to hold predictions
df_predictions = pd.DataFrame()

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

    # Predict the temperatures for this city using the respective model
    city_predictions = models[city].predict(X)

    # Store the predictions in the dataframe
    df_predictions[city] = city_predictions

# The target variable (temperature) remains the 'MAR' column from the dataset
target = y

# Create a stacking model with SVR models
stacked_model = StackingRegressor(
    estimators=[
        ('svr1', SVR(kernel='linear', C=100, epsilon=0.1)),
        ('svr2', SVR(kernel='linear', C=50, epsilon=0.2)),
        ('svr3', SVR(kernel='linear', C=200, epsilon=0.05))
    ],
    final_estimator=LinearRegression()
)

# Train the stacking model
stacked_model.fit(df_predictions, target)

# Evaluate the stacked model
stacked_predictions = stacked_model.predict(df_predictions)
stacked_mse = mean_squared_error(target, stacked_predictions)

print(f"Stacked Model MSE (SVR): {stacked_mse}")

# Save the stacked model
joblib.dump(stacked_model, 'models/combines/stacked_model_svr.pkl')

print("Stacked model trained and saved successfully!")
