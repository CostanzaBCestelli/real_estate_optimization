# src/forecasting.py

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path):
    """
    Load historical data from a CSV file.
    """
    data = pd.read_csv(file_path)
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Year', inplace=True)
    pivot_data = data.pivot_table(values='Total Return', index='Year', columns=['Sector', 'Region'])
    pivot_data.dropna(axis=1, inplace=True)
    return pivot_data

def hybrid_forecast(data, sector, region, forecast_years=7):
    """
    Perform hybrid forecasting for a given sector and region.
    """
    series = data[(sector, region)].dropna()

    # Step 1: Apply SARIMA or ARIMA Model
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = model.fit(disp=False)
        forecast = sarima_fit.forecast(steps=forecast_years)
    except:
        model = ARIMA(series, order=(1, 1, 1))
        arima_fit = model.fit()
        forecast = arima_fit.forecast(steps=forecast_years)

    # Step 2: Apply Forward-Looking Adjustments
    if sector == 'Office':
        gdp_growth = 0.014 if region == 'Europe' else 0.016
        inflation = 0.020 if region in ['Europe', 'Asia Pacific'] else 0.024
        sector_specific_adjustment = 0.012 if region == 'US' else 0.008
    elif sector == 'Industrial':
        gdp_growth = 0.018 if region == 'Asia Pacific' else 0.015
        inflation = 0.021 if region == 'US' else 0.019
        sector_specific_adjustment = 0.015 if 'ecommerce' in series.name else 0.010
    elif sector == 'Hotel':
        gdp_growth = 0.013 if region == 'Europe' else 0.017
        inflation = 0.023
        sector_specific_adjustment = 0.005 if region == 'UK' else 0.007
    else:
        gdp_growth = 0.016
        inflation = 0.022
        sector_specific_adjustment = 0.005

    adjusted_forecast = forecast + gdp_growth + inflation + sector_specific_adjustment
    historical_volatility = series.std()
    smoothing_factor = 0.6
    desmoothed_volatility = historical_volatility / (1 - smoothing_factor)

    results = {
        'Sector': sector,
        'Region': region,
        'Forecast': adjusted_forecast.values,
        'Historical Volatility': historical_volatility,
        'De-smoothed Volatility': desmoothed_volatility
    }
    return results


# Generate Forecast for Each Sector/Region Combination
# The function is applied to generate forecasts for all combinations of sectors and regions.
forecast_results = []
sectors = pivot_data.columns.get_level_values(0).unique()
regions = pivot_data.columns.get_level_values(1).unique()

for sector in sectors:
    for region in regions:
        if (sector, region) in pivot_data.columns:
            result = hybrid_forecast(pivot_data, sector, region)
            forecast_results.append(result)

# Convert Results to DataFrame
# The forecast results are compiled into a DataFrame for easy analysis and saving.
forecast_df = pd.DataFrame(forecast_results)

# Save the Forecast Results to CSV
# The results are saved to a CSV file for further use.
forecast_df.to_csv('/path/to/forecast_results.csv', index=False)

# Display Summary of Results
print(forecast_df.head())

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load Forecast Results
data = pd.read_csv('/path/to/forecast_results.csv')

# Extract Expected Returns and Volatility
data['Risk'] = data['De-smoothed Volatility']
data['Return'] = data['Forecast'].apply(lambda x: np.mean(eval(x)))
data = data[['Sector', 'Region', 'Return', 'Risk']]
