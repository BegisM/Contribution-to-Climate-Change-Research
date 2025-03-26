import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

cities = ['berlin', 'beijing', 'moscow', 'los_angeles',
          'new_york', 'sydney', 'tokyo', 'london',
          'buenos_Aires', 'mexico']

# Function to calculate AIC
def calculate_aic(n, sse, k):
    return n * np.log(sse / n) + 2 * k

# Function to calculate BIC
def calculate_bic(n, sse, k):
    return n * np.log(sse / n) + k * np.log(n)

best_results = {}

for city in cities:
    if city == 'london':
        df_part1 = pd.read_csv('weather/london_1960.csv')
        df_part2 = pd.read_csv('weather/london_1960-.csv')

        df_part1.replace(999.90, pd.NA, inplace=True)
        df_part2.replace(999.90, pd.NA, inplace=True)

        df_part1 = df_part1[(df_part1['YEAR'] >= 1900) & (df_part1['YEAR'] <= 1969)]
        df_part2 = df_part2[(df_part2['YEAR'] >= 1960)]

        df_part1.dropna(subset=['MAR'], inplace=True)
        df_part2.dropna(subset=['MAR'], inplace=True)

        df_part1 = df_part1[df_part1['YEAR'] < 1960]
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

    n = len(X)  # Number of data points

    best_aic = float('inf')
    best_bic = float('inf')
    best_degree_aic = None
    best_degree_bic = None

    for degree in range(1, 11):  # Try polynomial degrees from 1 to 10
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        sse = np.sum((y - y_pred) ** 2)

        k = degree + 1  # Number of parameters

        aic = calculate_aic(n, sse, k)
        bic = calculate_bic(n, sse, k)

        if aic < best_aic:
            best_aic = aic
            best_degree_aic = degree

        if bic < best_bic:
            best_bic = bic
            best_degree_bic = degree

    best_results[city] = {
        "Best Degree (AIC)": best_degree_aic,
        "Best AIC": best_aic,
        "Best Degree (BIC)": best_degree_bic,
        "Best BIC": best_bic
    }

# Print results
for city, results in best_results.items():
    print(f"{city.capitalize()} - Best Degree (AIC): {results['Best Degree (AIC)']}, Best AIC: {results['Best AIC']}")
    print(f"{city.capitalize()} - Best Degree (BIC): {results['Best Degree (BIC)']}, Best BIC: {results['Best BIC']}")
    print("-" * 50)
