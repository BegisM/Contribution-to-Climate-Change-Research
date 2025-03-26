import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../weather/berlin.csv')

# Replace 999.90 with NaN (Not a Number)
data.replace(999.90, pd.NA, inplace=True)

# Drop rows where 'MAR' (March) has missing values
march_data = data[(data['YEAR'] >= 1900) & (data['YEAR'] <= 2000)].dropna(subset=['MAR'])

# Check if there are negative temperatures in the data
print(f"Min Temp: {march_data['MAR'].min()}, Max Temp: {march_data['MAR'].max()}")

# Select the 'YEAR' and 'MAR' (March) columns
X = march_data[['YEAR']]  # Features (Year)
y = march_data['MAR']     # Target (March Temperature)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict the temperature for future years (e.g., 2001-2020)
future_years = pd.DataFrame({'YEAR': range(2001, 2021)})
future_predictions = model.predict(future_years)

# Print predictions for future years
print("Predicted March temperatures for 2001-2020:")
for year, temp in zip(future_years['YEAR'], future_predictions):
    print(f"Year {year}: {temp:.2f}°C")

# Visualize the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Year')
plt.ylabel('March Temperature (°C)')
plt.title('Temperature Trend for March (1900-2000) and Predictions for Future Years')

# Adjust y-axis limits to include negative temperatures
plt.ylim(min(y.min(), future_predictions.min()) - 2, max(y.max(), future_predictions.max()) + 2)

plt.legend()
plt.show()
