# Time Series Forecasting Hub 📈
A comprehensive retail sales forecasting project comparing four different modeling approaches to predict daily sales across multiple stores and products.

## Overview
This project analyzes daily retail sales data from 7 stores and 10 products (70 combinations total) to determine the best forecasting approach. It compares individual vs. clustered modeling strategies using AutoReg, SARIMA, and Prophet models with comprehensive performance evaluation.

## Features

### Data Analysis Functions
- **Rolling Statistics**: Multi-timeframe trend analysis (7, 90, 180, 365 days)
- **Stationarity Testing**: Augmented Dickey-Fuller tests for time series properties
- **Seasonal Decomposition**: Separates trend, seasonal, and residual components
- **Autocorrelation Analysis**: ACF/PACF plots for pattern identification

### Machine Learning Pipeline
- **Feature Engineering**: Advanced hierarchical differencing for stationarity
- **Model Comparison**: Four different forecasting approaches:
  - Individual AutoReg models (70 models)
  - Individual SARIMA models (70 models) 
  - Clustered SARIMA models (17 models)
  - Prophet models (70 models)
- **Performance Metrics**: RMSE, MAPE, and success rate tracking
- **Model Evaluation**: Out-of-sample testing on full year of data

### Data Visualization
- Rolling statistics and trend analysis plots
- Seasonal decomposition visualizations
- Store-level performance comparisons
- Model accuracy rankings and insights

## Dependencies
- pandas>=2.3.1
- numpy>=2.3.2
- matplotlib>=3.10.3
- seaborn>=0.13.2
- statsmodels
- scikit-learn
- prophet

## Usage
- **Data Structure**: Requires `train.csv` (2017-2018) and `test.csv` (2019) with Date, store, product, number_sold columns
- **Execution**: Run `python main.py` to perform complete analysis
- **Output**: Performance metrics, visualizations, and model comparison results

## Key Insights
The project reveals trade-offs between model complexity and accuracy, showing when to use individual models (high accuracy) vs. clustered models (operational efficiency) vs. Prophet (business-friendly) depending on your specific needs.

## Results
- **Performance Rankings**: Identifies best modeling approach for your data
- **Store Insights**: Highlights which stores/products are easier to forecast
- **Business Recommendations**: Optimal strategy based on accuracy vs. complexity trade-offs
- **Comprehensive Metrics**: Detailed RMSE/MAPE analysis across all approaches

Perfect for data scientists, retail analysts, or anyone interested in time series forecasting! 🚀📊

## 📊 Expected Results

The analysis provides:
1. **Performance Rankings**: Which modeling approach works best for your data
2. **Store Insights**: Which stores/products are easier/harder to forecast
3. **Complexity Analysis**: Computational vs. accuracy trade-offs
4. **Business Recommendations**: Optimal modeling strategy for different use cases
