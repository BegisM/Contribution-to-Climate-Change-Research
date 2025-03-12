import pandas as pd
import joblib  # For saving models
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline  # For combining polynomial transformation and model

cities = ['berlin', 'beijing', 'moscow', 'los_angeles',
          'new_york', 'sydney', 'tokyo', 'london',
          'buenos_Aires', 'mexico']

models = {}

# Degree of the polynomial
degree = 3
epsilon_value = 1.4  # epsilon value for SVR

for city in cities:
    if city == 'london':
        df_part1 = pd.read_csv('weather/london_1960.csv')
        df_part2 = pd.read_csv('weather/london_1960-.csv')

        # Replace missing values (e.g., 999.90) with NaN and drop rows with missing temperature data
        df_part1.replace(999.90, pd.NA, inplace=True)
        df_part2.replace(999.90, pd.NA, inplace=True)

        # Filter the data to keep only the years in the range 1900-2000 for part1 and 1960-now for part2
        df_part1 = df_part1[(df_part1['YEAR'] >= 1900) & (df_part1['YEAR'] <= 1969)]
        df_part2 = df_part2[(df_part2['YEAR'] >= 1960)]  # Keep the full range of years

        # Remove rows where temperature is missing
        df_part1.dropna(subset=['MAR'], inplace=True)
        df_part2.dropna(subset=['MAR'], inplace=True)

        df_part1 = df_part1[df_part1['YEAR'] < 1960]  # Remove 1960-1969 from part1

        # Now combine the two datasets
        df = pd.concat([df_part1, df_part2])
        df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2000)].dropna(subset=['MAR'])

    elif city == 'mexico':
        df = pd.read_csv(f'weather/mexico_1921.csv')
        df.replace(999.90, pd.NA, inplace=True)
        df = df[(df['YEAR'] >= 1921) & (df['YEAR'] <= 2021)].dropna(subset=['MAR'])

    else:
        df = pd.read_csv(f'weather/{city}.csv')
        df.replace(999.90, pd.NA, inplace=True)
        df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2000)].dropna(subset=['MAR'])

    X = df[['YEAR']]
    y = df['MAR']

    # Create a pipeline that combines PolynomialFeatures and SVR (with polynomial kernel)
    model = make_pipeline(PolynomialFeatures(degree=degree),
                          SVR(kernel='poly', degree=degree, epsilon=epsilon_value))

    # Train the model
    model.fit(X, y)

    models[city] = model

    # Save the complete model (polynomial transformer + SVR model)
    joblib.dump(model, f'models/svr_polynomial_regression/{city}_svr_polynomial_model.pkl')

print("SVR with polynomial kernel models trained and saved successfully!")
