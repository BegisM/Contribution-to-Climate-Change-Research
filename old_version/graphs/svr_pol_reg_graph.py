import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('../weather/moscow.csv')

# Replace 999.90 with NaN (missing data)
data.replace(999.90, pd.NA, inplace=True)

# Drop rows where 'MAR' (March) has missing values
march_data = data[(data['YEAR'] >= 1900) & (data['YEAR'] <= 2000)].dropna(subset=['MAR'])

# Extract Year and March Temperature
X = march_data[['YEAR']].values  # Convert to NumPy array
y = march_data['MAR'].values

# Standardize the features (important for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()  # Flatten y after scaling

# Train SVR models with a polynomial kernel
epsilon = 1.4  # Wider tube for chaotic data
svr = SVR(kernel='poly', degree=3, C=100, epsilon=epsilon)
svr.fit(X_scaled, y_scaled)

# Predict temperatures for 1900-2020
X_future = np.array(range(1900, 2021)).reshape(-1, 1)
X_future_scaled = scaler_X.transform(X_future)
y_future_scaled = svr.predict(X_future_scaled)
y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1))  # Convert back

# Calculate upper and lower bounds of the ε-tube
y_tube_upper = y_future + epsilon  # Upper bound
y_tube_lower = y_future - epsilon  # Lower bound

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_future, y_future, color='red', label='SVR Prediction')

# Plot the ε-tube with a larger width
plt.fill_between(X_future.ravel(), y_tube_lower.ravel(), y_tube_upper.ravel(),
                 color='gray', alpha=0.3, label=f'ε-Tube (ε={epsilon})')

plt.xlabel('Year')
plt.ylabel('March Temperature (°C)')
plt.title(f'SVR Temperature Trend with ε={epsilon}')
plt.legend()
plt.show()
