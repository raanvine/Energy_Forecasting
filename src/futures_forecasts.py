import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def forecast_arima(df, value_col, date_col='date', forecast_periods=180):
    df = df[[date_col, value_col]].copy()
    df = df.groupby(date_col).mean().reset_index()
    df.set_index(date_col, inplace=True)
    df = df.asfreq('D')
    df[value_col].interpolate(method='time', inplace=True)
    model = ARIMA(df[value_col], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_df = forecast.summary_frame(alpha=0.05)
    forecast_df.rename(columns={
        'mean': 'forecast',
        'mean_ci_lower': 'lower_ci',
        'mean_ci_upper': 'upper_ci'
    }, inplace=True)
    forecast_df.index.name = date_col
    forecast_df.reset_index(inplace=True)
    return forecast_df

def forecast_prophet(df, value_col, date_col='date', forecast_periods=180):
    prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'}).copy()
    prophet_df = prophet_df.groupby('ds').mean().reset_index()
    model = Prophet(interval_width=0.95)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_df.rename(columns={
        'ds': date_col,
        'yhat': 'forecast',
        'yhat_lower': 'lower_ci',
        'yhat_upper': 'upper_ci'
    }, inplace=True)
    return forecast_df

def plot_forecasts(original_df, arima_forecast_df, prophet_forecast_df, value_col, date_col='date', title='Forecast', filename=None):
    plt.figure(figsize=(14, 7))
    plt.plot(original_df[date_col], original_df[value_col], label='Historical', color='black')
    plt.plot(arima_forecast_df[date_col], arima_forecast_df['forecast'], label='ARIMA Forecast', color='blue')
    plt.fill_between(arima_forecast_df[date_col], arima_forecast_df['lower_ci'], arima_forecast_df['upper_ci'], color='blue', alpha=0.2)
    plt.plot(prophet_forecast_df[date_col], prophet_forecast_df['forecast'], label='Prophet Forecast', color='orange')
    plt.fill_between(prophet_forecast_df[date_col], prophet_forecast_df['lower_ci'], prophet_forecast_df['upper_ci'], color='orange', alpha=0.2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def run_forecasting_pipeline(csv_file, value_col, output_name):
    df = pd.read_csv(f'data/{csv_file}')
    df['date'] = pd.to_datetime(df['date'])
    arima_forecast = forecast_arima(df, value_col)
    prophet_forecast = forecast_prophet(df, value_col)
    os.makedirs('data', exist_ok=True)
    arima_forecast.to_csv(f'data/{output_name}_arima_forecast.csv', index=False)
    prophet_forecast.to_csv(f'data/{output_name}_prophet_forecast.csv', index=False)
    os.makedirs('reports', exist_ok=True)
    plot_forecasts(
        df, arima_forecast, prophet_forecast, value_col,
        title=f'{output_name} Forecasts (ARIMA & Prophet)',
        filename=f'reports/{output_name}_forecast_plot.png'
    )
    print(f"Forecasting complete for {output_name}. CSVs saved to 'data/'. Plots saved to 'reports/'.")

if __name__ == '__main__':
    run_forecasting_pipeline('futures_pricing.csv', 'value', 'futures_prices')
