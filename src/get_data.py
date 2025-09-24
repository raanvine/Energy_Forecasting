import os
import requests
import pandas as pd
from config import API_KEY, BASE_URL_FUTURESSPOT

def get_natural_gas_futuresspot(start_date='2019-01-01', end_date='2024-12-31', length=5000):
    params = {
        'api_key': API_KEY,
        'frequency': 'daily',
        'data[0]': 'value',
        'start': start_date,
        'end': end_date,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': 0,
        'length': length,
    }

    response = requests.get(BASE_URL_FUTURESSPOT, params=params)
    response.raise_for_status()
    json_data = response.json()

    # Extract data list 
    data_points = json_data.get('response', {}).get('data', [])
    df = pd.DataFrame(data_points)

    # Convert 'period' to datetime and rename to 'date'
    df['date'] = pd.to_datetime(df['period'])

    # reorder columns so 'date' is first
    cols = ['date'] + [col for col in df.columns if col != 'date' and col != 'period']
    df = df[cols]

    # Sort by increasing date 
    df = df.sort_values('date').reset_index(drop=True)

    return df

if __name__ == '__main__':
    df = get_natural_gas_futuresspot()
    print("Sample of combined futures and spot data:")
    print(df.head())

    # Ensure data folder exists
    os.makedirs('data', exist_ok=True)

    # Save the full dataset to CSV
    csv_path = 'data/combined_natural_gas_futures_spot.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
