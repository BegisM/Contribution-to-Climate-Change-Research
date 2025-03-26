import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data
data = pd.read_csv('../weather/berlin.csv')

# Replace 999.90 with NaN (missing data)
data.replace(999.90, pd.NA, inplace=True)

# Drop rows where 'MAR' (March) has missing values
march_data = data[(data['YEAR'] >= 1900) & (data['YEAR'] <= 2000)].dropna(subset=['MAR'])

# Extract Year and March Temperature
X = march_data[['YEAR']].values  # Convert to NumPy array
y = march_data['MAR'].values

# Create polynomial features of degree 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict temperatures for 1900-2020
X_future = np.array(range(1900, 2021)).reshape(-1, 1)  # Future years
X_future_poly = poly.transform(X_future)  # Transform future years into polynomial features
y_future = model.predict(X_future_poly)  # Predict temperatures

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')  # Scatter plot of actual data
plt.plot(X_future, y_future, color='green', label='Polynomial Regression (Degree=3)')  # Polynomial fit

plt.xlabel('Year')
plt.ylabel('March Temperature (°C)')
plt.title('Polynomial Regression (Degree 3) - Berlin March Temperature')
plt.legend()
plt.show()
