import pandas as pd
import numpy as np
from normalization import z_normalization


def get_data(start, end):
    # Read the data from the CSV file
    df = pd.read_csv('Weather/Postdam.csv')

    # Convert 'DATE' to datetime format
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Extract year, month, and day
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['MONTH_DAY'] = df['DATE'].dt.strftime('%m-%d')  # Keep only MM-DD for plotting consistency

    # Adjust winter season year
    df['SEASON_YEAR'] = df['YEAR']
    df.loc[df['MONTH'] < 10, 'SEASON_YEAR'] -= 1  # Assign Jan-Apr to thew previous year's winter season

    # Filter for the winter periods between 1975 and 2000
    df_filtered = df[(df['SEASON_YEAR'].between(start, end)) & (df['MONTH'].isin([10, 11, 12, 1, 2, 3, 4]))]

    return df_filtered


def get_day_of_year(month, day):
    """Returns the correct day number in the non-leap year calendar."""
    days_in_months = {1: 0, 2: 31, 3: 59, 4: 90,  # Jan-Apr (new year, starts from 1)
                      10: 273, 11: 304, 12: 334}  # Oct-Dec (previous year, starts from 274)

    return days_in_months[month] + day

def get_month_of_year(day_of_year):
    """Returns the correct month-day pair in the non-leap year calendar by the number of that day."""
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    month = 1
    while day_of_year > days_in_months[month - 1]:
        day_of_year -= days_in_months[month - 1]
        month += 1

    return month, day_of_year


def get_all_data(start, end, temp, maximum=False, minumum=False):
    data = get_data(start, end)

    # Remove Feb 29 to standardize across all years
    data = data[data['MONTH_DAY'] != '02-29']

    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns

    daily_avg = data.groupby("MONTH_DAY")[temp].mean().reset_index()

    # Group by MONTH_DAY to get the appropriate temperature per day
    if maximum:
        daily_avg = data.groupby("MONTH_DAY")[temp].max().reset_index()

    elif minumum:
        daily_avg = data.groupby("MONTH_DAY")[temp].min().reset_index()

    # Extract month and day from MONTH_DAY
    daily_avg["MONTH"] = daily_avg["MONTH_DAY"].str[:2].astype(int)
    daily_avg["DAY"] = daily_avg["MONTH_DAY"].str[3:].astype(int)

    # Compute correct day index
    daily_avg["DAY_INDEX"] = daily_avg.apply(lambda row: get_day_of_year(row["MONTH"], row["DAY"]), axis=1)

    # Sort by DAY_INDEX (important for model training)
    daily_avg = daily_avg.sort_values(by=["DAY_INDEX"], key=lambda x: np.where(x < 274, x + 365, x)).reset_index(
        drop=True)

    return daily_avg



def get_training_data(daily_avg, temp):
    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_columns', None)  # Show all columns
    # print(daily_avg)

    # Create X (DAY_INDEX and its square) and Y (average temp)
    x, x_mu, x_sigma = z_normalization(np.array(daily_avg.index + 1).reshape(-1, 1))
    x_norm = np.hstack((x, x ** 2))
    y = np.array(daily_avg[temp]).reshape(-1, 1)

    # # Normalize data
    # x_norm, x_mu, x_sigma = z_normalization(x)
    y_norm, y_mu, y_sigma = z_normalization(y)

    return (x_norm, x_mu, x_sigma), (y_norm, y_mu, y_sigma)


def get_month_day(daily_avg, index):
    return daily_avg.loc[index, 'MONTH'], daily_avg.loc[index, 'DAY']

def get_single_season_data(start_year):
    """
    Get weather data for a single winter season from October of `start_year`
    to April of `start_year + 1`, without averaging and preserving original order.
    Resets the DataFrame index to start from 0.
    """
    # Read the data
    df = pd.read_csv('Weather/Postdam.csv')

    # Convert 'DATE' to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Extract year and month
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month

    # Filter for Oct-Dec of start_year and Jan-Apr of start_year + 1
    mask = (
        ((df['YEAR'] == start_year) & (df['MONTH'].isin([10, 11, 12]))) |
        ((df['YEAR'] == start_year + 1) & (df['MONTH'].isin([1, 2, 3, 4])))
    )

    df_filtered = df[mask].copy()

    # Remove leap day if needed (Feb 29)
    df_filtered['MONTH_DAY'] = df_filtered['DATE'].dt.strftime('%m-%d')
    df_filtered = df_filtered[df_filtered['MONTH_DAY'] != '02-29']

    # Reset index without changing order
    df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered

