# README: Hybrid Forecasting and Portfolio Optimization Script

## Overview

This script provides a comprehensive framework for forecasting Real Estate financial returns across various sectors and regions using a hybrid methodology. It integrates statistical time-series models (SARIMA and ARIMA) with forward-looking adjustments to account for macroeconomic factors, sector-specific trends, and regional influences. Additionally, it includes a Michaud resampling process for portfolio optimization based on forecasted returns and risks.

---

## Features

1. **Hybrid Forecasting Methodology**:
   - Combines SARIMA (for seasonality) and ARIMA models for time-series forecasting.
   - Applies forward-looking adjustments for macroeconomic factors (e.g., GDP growth, inflation).

2. **Volatility Analysis**:
   - Calculates historical volatility for each sector-region combination.
   - De-smooths volatility to provide a realistic risk estimate.

3. **Portfolio Optimization**:
   - Uses Michaud resampling to create a robust and diversified portfolio.
   - Ensures optimal allocation with risk constraints and maximum weight limits.

4. **Output**:
   - Generates sector-region-specific forecasts and risk metrics.
   - Provides an optimised portfolio allocation.

---

## Installation and Requirements

### Dependencies

The script requires the following Python libraries:
- `numpy`
- `pandas`
- `statsmodels`
- `scipy`

Install dependencies using:
```bash
pip install numpy pandas statsmodels scipy
