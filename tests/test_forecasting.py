# tests/test_forecasting.py

import pandas as pd
import numpy as np
import os
from src.forecasting import hybrid_forecast, load_data

def test_hybrid_forecast_with_synthetic_data():
    # Construct the path to the synthetic data
    test_dir = os.path.dirname(__file__)
    data_path = os.path.join(test_dir, 'data', 'synthetic_historical_data.csv')
    
    # Load the synthetic historical data using the refactored load_data function
    pivot_data = load_data(data_path)

    # Pick a known sector-region combination from synthetic data
    sector = 'Office'
    region = 'Europe'

    # Run hybrid forecasting for a 7-year horizon
    result = hybrid_forecast(pivot_data, sector, region, forecast_years=7)

    # Assertions:
    # 1. Check forecast length
    assert len(result['Forecast']) == 7, "Forecast should have 7 values representing 7 years."

    # 2. No NaN values in the forecast
    assert not np.isnan(result['Forecast']).any(), "Forecast should not contain NaN values."

    # 3. Historical Volatility should be a finite number
    assert np.isfinite(result['Historical Volatility']), "Historical volatility should be a finite number."

    # 4. De-smoothed Volatility should be greater than Historical Volatility if smoothing_factor < 1
    assert result['De-smoothed Volatility'] > result['Historical Volatility'], \
        "De-smoothed volatility should be greater than historical volatility if smoothing factor < 1."

    # 5. Check forecast values are within a reasonable range
    # Given the data, returns likely range within a certain plausible band. Adjust as needed.
    # Example: forecasted annual returns should not be outside -50% to +100%
    assert (result['Forecast'] > -0.5).all(), "Forecast values should not be unrealistically low."
    assert (result['Forecast'] < 1.0).all(), "Forecast values should not be unrealistically high."

    # If all assertions pass, the test will succeed.

