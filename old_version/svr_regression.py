import pandas as pd
import joblib  # For saving models
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

cities = ['berlin', 'beijing', 'moscow', 'los_angeles',
          'new_york', 'sydney', 'tokyo', 'london',
          'buenos_Aires', 'mexico']
models = {}

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

    X = df[['YEAR']].values  # Use YEAR as the feature
    y = df['MAR'].values  # Use MAR (March temperature) as the target variable

    # Standardize the features (important for SVR)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()  # Flatten y after scaling

    # Train SVR model with a polynomial kernel
    epsilon = 1.4  # Adjust this parameter based on the data
    svr = SVR(kernel='linear', degree=3, C=100, epsilon=epsilon)
    svr.fit(X_scaled, y_scaled)

    # Save the model
    models[city] = svr
    joblib.dump(svr, f'models/svr_lr/{city}_svr_model.pkl')

print("SVR models trained and saved successfully!")
