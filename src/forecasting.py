import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import random

# Load Data
# Assuming the data is loaded from a CSV containing columns: 'Year', 'Sector', 'Region', 'Total Return'
data = pd.read_csv('/path/to/historical_data.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Pivot the data for analysis
pivot_data = data.pivot_table(values='Total Return', index='Year', columns=['Sector', 'Region'])
pivot_data.dropna(axis=1, inplace=True)

# Function to perform Hybrid Forecasting Methodology for Each Sector/Region

def hybrid_forecast(data, sector, region, forecast_years=7):
    # Extract time series data for specific sector/region
    series = data[(sector, region)].dropna()

    # Step 1: Apply SARIMA or ARIMA Model
    try:
        # SARIMA Model for sectors/regions with potential seasonality
        # Example: Sectors like 'Retail' or 'Hotel' in regions with known seasonal influences are better modeled using SARIMA
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = model.fit(disp=False)
        forecast = sarima_fit.forecast(steps=forecast_years)
    except:
        # Fallback to ARIMA if SARIMA fails
        # ARIMA is used for sectors/regions where seasonality is not pronounced, such as 'Office' or 'Industrial'
        model = ARIMA(series, order=(1, 1, 1))
        arima_fit = model.fit()
        forecast = arima_fit.forecast(steps=forecast_years)

    # Step 2: Apply Forward-Looking Adjustments
    # Forward-looking adjustments account for expected macroeconomic factors, such as GDP growth, inflation, and sector-specific trends.
    # These adjustments differ based on the sector and region being forecasted.

    # Example adjustments based on region and sector
    if sector == 'Office':
        gdp_growth = 0.014 if region == 'Europe' else 0.016  # GDP growth rates vary by region
        inflation = 0.020 if region in ['Europe', 'Asia Pacific'] else 0.024  # Different inflation rates for different regions
        sector_specific_adjustment = 0.012 if region == 'US' else 0.008  # Prime location or regional premiums for Office
    elif sector == 'Industrial':
        gdp_growth = 0.018 if region == 'Asia Pacific' else 0.015
        inflation = 0.021 if region == 'US' else 0.019
        sector_specific_adjustment = 0.015 if 'ecommerce' in series.name else 0.010  # E-commerce-driven adjustments
    elif sector == 'Hotel':
        gdp_growth = 0.013 if region == 'Europe' else 0.017
        inflation = 0.023
        sector_specific_adjustment = 0.005 if region == 'UK' else 0.007  # Tourist season adjustments
    else:
        gdp_growth = 0.016  # Default value for other sectors
        inflation = 0.022
        sector_specific_adjustment = 0.005  # Generic sector adjustment

    # Apply forward-looking adjustments
    adjusted_forecast = forecast + gdp_growth + inflation + sector_specific_adjustment

    # Step 3: Calculate Historical Volatility
    # Volatility is calculated to understand the historical risk associated with each sector/region.
    historical_volatility = series.std()

    # Step 4: De-smooth Volatility to Get a More Realistic Risk Estimate
    # De-smoothing volatility helps to account for the smoothing effect inherent in less frequently traded assets, such as real estate.
    smoothing_factor = 0.6
    desmoothed_volatility = historical_volatility / (1 - smoothing_factor)

    # Compile Results
    # The results include the forecasted returns, historical volatility, and de-smoothed volatility for each sector and region.
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
