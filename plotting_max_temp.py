import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('Weather/temperature_data.csv')

# Convert 'DATE' to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Extract year, month, and day
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month
df['MONTH_DAY'] = df['DATE'].dt.strftime('%m-%d')  # Keep only MM-DD for plotting consistency

# Adjust the winter season year
df['SEASON_YEAR'] = df['YEAR']
df.loc[df['MONTH'] < 10, 'SEASON_YEAR'] -= 1  # Assign Jan-Apr to thew previous year's winter season

# Filter for the winter periods between 1975 and 2000
# df_filtered = df[(df['SEASON_YEAR'].between(1975, 2000)) & (df['MONTH'].isin([10, 11, 12, 1, 2, 3, 4]))]
df_filtered_later = df[(df['SEASON_YEAR'].between(2000, 2025)) & (df['MONTH'].isin([10, 11, 12, 1, 2, 3, 4]))]

# Plot the scatter plot
plt.figure(figsize=(12, 6))
# plt.scatter(df_filtered['MONTH_DAY'], df_filtered['TMIN'], color='red', alpha=0.5, label='Max Temperature (TMAX)')
plt.scatter(df_filtered_later['MONTH_DAY'], df_filtered_later['TMIN'], color='blue', alpha=0.3, label='Max Temperature (TMAX)')

# Labels and title
plt.xlabel('Date (Month)')
plt.ylabel('Temperature (°C)')
plt.title('Max Temperature from October to April (1975-2000)')
plt.legend()

# Set x-axis ticks to only show months
months = ['10-01', '11-01', '12-01', '01-01', '02-01', '03-01', '04-01']
month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
plt.xticks(months, month_labels)

# Display the plot
plt.tight_layout()
plt.show()
