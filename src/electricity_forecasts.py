import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# === CONFIG ===
CSV_FILE = 'data/WholesaleElectricity2020-23.csv'
VALUE_COL = 'Wtd avg price $/MWh'
DATE_COL = 'Trade date'
OUTPUT_NAME = 'wholesale_electricity'

# === FUNCTIONS ===

def forecast_arima(df, value_col, forecast_periods=180):
    df = df.copy()
    df.set_index('date', inplace=True)
    df = df.asfreq('D')  # daily frequency
    df[value_col].interpolate(method='time', inplace=True)

    model = ARIMA(df[value_col], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_df = forecast.summary_frame(alpha=0.05)

    forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
    forecast_df.rename(columns={
        'mean': 'forecast',
        'mean_ci_lower': 'lower_ci',
        'mean_ci_upper': 'upper_ci'
    }, inplace=True)
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'date'}, inplace=True)
    return forecast_df

def forecast_prophet(df, value_col, forecast_periods=180):
    prophet_df = df[['date', value_col]].rename(columns={'date': 'ds', value_col: 'y'}).copy()
    model = Prophet(interval_width=0.95)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_df.rename(columns={
        'ds': 'date',
        'yhat': 'forecast',
        'yhat_lower': 'lower_ci',
        'yhat_upper': 'upper_ci'
    }, inplace=True)
    return forecast_df

def plot_forecasts(original_df, arima_df, prophet_df, value_col, title, filename):
    plt.figure(figsize=(14, 7))
    plt.plot(original_df['date'], original_df[value_col], label='Historical', color='black')

    plt.plot(arima_df['date'], arima_df['forecast'], label='ARIMA Forecast', color='blue')
    plt.fill_between(arima_df['date'], arima_df['lower_ci'], arima_df['upper_ci'], color='blue', alpha=0.2)

    plt.plot(prophet_df['date'], prophet_df['forecast'], label='Prophet Forecast', color='orange')
    plt.fill_between(prophet_df['date'], prophet_df['lower_ci'], prophet_df['upper_ci'], color='orange', alpha=0.2)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# === MAIN PIPELINE ===

def run_pipeline():
    df = pd.read_csv(CSV_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df[[DATE_COL, VALUE_COL]].copy()
    df.rename(columns={DATE_COL: 'date'}, inplace=True)
    df = df.groupby('date').mean().reset_index()

    arima_df = forecast_arima(df, VALUE_COL)
    prophet_df = forecast_prophet(df, VALUE_COL)

    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    arima_df.to_csv(f'data/{OUTPUT_NAME}_arima_forecast.csv', index=False)
    prophet_df.to_csv(f'data/{OUTPUT_NAME}_prophet_forecast.csv', index=False)

    plot_forecasts(
        df, arima_df, prophet_df, VALUE_COL,
        title='Wholesale Electricity Price Forecast (ARIMA & Prophet)',
        filename='reports/electricity_forecast_plot.png'
    )

    print("Forecasts complete. CSVs saved to 'data/', plot saved to 'reports/'.")

# === RUN ===
if __name__ == '__main__':
    run_pipeline()