import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load the training and testing datasets
time_series_data_train = pd.read_csv('train.csv')
time_series_data_test = pd.read_csv('test.csv')

# Display basic information about the datasets
print(time_series_data_train.info())
print(time_series_data_test.info())

# Convert Date columns to datetime format
time_series_data_train['Date'] = pd.to_datetime(time_series_data_train['Date'], format='%Y-%m-%d')
time_series_data_test['Date'] = pd.to_datetime(time_series_data_test['Date'], format='%Y-%m-%d')

print(time_series_data_train.info())

# Extract data for Store 0, Product 0 for initial analysis
tsd_train_1 = time_series_data_train.loc[(time_series_data_train['store'] == 0) & (time_series_data_train['product'] == 0)].copy()
tsd_test_1 = time_series_data_test.loc[(time_series_data_test['store'] == 0) & (time_series_data_test['product'] == 0)].copy()

print(tsd_train_1.info())

# Set Date as index and ensure daily frequency
tsd_train_1.set_index('Date', inplace=True)
tsd_train_1 = tsd_train_1.asfreq('D')  # Ensure daily frequency
tsd_test_1.set_index('Date', inplace=True)
tsd_test_1 = tsd_test_1.asfreq('D')  # Ensure daily frequency

# Plot the time series data to visualize patterns
plt.figure(figsize=(20, 10))
plt.plot(tsd_train_1['number_sold'], color='blue', label='Train')
plt.plot(tsd_test_1['number_sold'], color='red', label='Test')
plt.title('Time Series Plot: Store 0, Product 0')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()

# Import required libraries for time series analysis
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

# =============================================================================
# EXPLORATORY DATA ANALYSIS - ROLLING STATISTICS
# =============================================================================

# Plot rolling means to identify trends and seasonality patterns
plt.figure(figsize=(20, 10))
tsd_train_1['number_sold'].plot(label='Original Data')
tsd_train_1['number_sold'].rolling(7).mean().plot(label='7-Day Rolling Mean (Weekly)')
tsd_train_1['number_sold'].rolling(90).mean().plot(label='90-Day Rolling Mean (Quarterly)')
tsd_train_1['number_sold'].rolling(180).mean().plot(label='180-Day Rolling Mean (Semi-Annual)')
tsd_train_1['number_sold'].rolling(365).mean().plot(label='365-Day Rolling Mean (Annual)')
plt.title('Rolling Statistics Analysis - Trend Detection')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()

# Plot rolling means and standard deviations to check for stationarity
plt.figure(figsize=(20, 10))
tsd_train_1['number_sold'].plot(label='Original Data')
tsd_train_1['number_sold'].rolling(180).mean().plot(label='180-Day Rolling Mean')
tsd_train_1['number_sold'].rolling(365).mean().plot(label='365-Day Rolling Mean')
tsd_train_1['number_sold'].rolling(180).std().plot(label='180-Day Rolling Std Dev')
tsd_train_1['number_sold'].rolling(365).std().plot(label='365-Day Rolling Std Dev')
plt.axhline(tsd_train_1['number_sold'].mean(), linestyle='--', label='Overall Mean', color='black')
plt.title('Rolling Statistics - Mean vs Standard Deviation Analysis')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()

# =============================================================================
# STATIONARITY TESTING - AUGMENTED DICKEY-FULLER TEST
# =============================================================================

# Perform ADF test to check for unit root (non-stationarity)
result = adfuller(tsd_train_1['number_sold'])

print("=== AUGMENTED DICKEY-FULLER TEST RESULTS ===")
print(f'ADF Statistic: {result[0]:.6f}')
print(f'p-value: {result[1]:.6f}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value:.3f}')

# Interpret the p-value for stationarity
if result[1] <= 0.05:
    print("✓ The series is likely STATIONARY (reject null hypothesis)")
else:
    print("✗ The series is likely NON-STATIONARY (fail to reject null hypothesis)")

# =============================================================================
# AUTOCORRELATION ANALYSIS - ACF AND PACF PLOTS
# =============================================================================

# Plot ACF and PACF to identify correlation patterns and potential model parameters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(tsd_train_1['number_sold'], lags=365, ax=axes[0])
plot_pacf(tsd_train_1['number_sold'], lags=365, ax=axes[1])
axes[0].set_title('Autocorrelation Function (ACF)')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# =============================================================================
# SEASONAL DECOMPOSITION ANALYSIS
# =============================================================================

# Decompose the time series to identify trend, seasonal, and residual components
decomposition = seasonal_decompose(tsd_train_1['number_sold'], model='additive', period=365)
decomposition.plot()
plt.suptitle('Seasonal Decomposition - Additive Model (365-day period)', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# DIFFERENCING FOR STATIONARITY
# =============================================================================

# Apply seasonal differencing to remove seasonality and make series stationary
print("=== APPLYING SEASONAL DIFFERENCING ===")
tsd_train_1['diff'] = tsd_train_1['number_sold'] - tsd_train_1['number_sold'].shift(1)
tsd_train_1.dropna(inplace=True)

# Plot the differenced series
plt.figure(figsize=(20, 6))
plt.plot(tsd_train_1['diff'])
plt.title('First-Order Differenced Time Series (Trend Removal)')
plt.xlabel('Date')
plt.ylabel('Differenced Values')
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# STATIONARITY TEST ON DIFFERENCED DATA
# =============================================================================

# Perform ADF test on differenced data to confirm stationarity
result_diff = adfuller(tsd_train_1['diff'])
print("=== ADF TEST ON DIFFERENCED DATA ===")
print(f"ADF Statistic: {result_diff[0]:.6f}")
print(f"p-value: {result_diff[1]:.6f}")

if result_diff[1] <= 0.05:
    print("✓ Differenced series is STATIONARY")
else:
    print("✗ Differenced series is still NON-STATIONARY")

# Plot ACF and PACF for differenced data
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(tsd_train_1['diff'], lags=365, ax=axes[0])
plot_pacf(tsd_train_1['diff'], lags=365, ax=axes[1])
axes[0].set_title('ACF - Differenced Data')
axes[1].set_title('PACF - Differenced Data')
plt.tight_layout()
plt.show()

# Seasonal decomposition of differenced data
decomposition_diff = seasonal_decompose(tsd_train_1['diff'], model='additive', period=365)
decomposition_diff.plot()
plt.suptitle('Seasonal Decomposition - Differenced Data', fontsize=16)
plt.tight_layout()
plt.show()



# =============================================================================
# MODEL PREPARATION AND INITIALIZATION
# =============================================================================
# Suppress specific warnings to clean up output during model fitting
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# Import evaluation metrics for model performance assessment
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error

# Initialize result storage lists for all modeling approaches
# Each list will store performance metrics for comparison across methods
results_list_AR = []                    # AutoReg individual models (70 models)
results_list_SARIMA_individual = []     # SARIMA individual models (70 models)  
results_list_SARIMA_cluster = []        # SARIMA clustered models (17 models)
results_list_prophet = []               # Prophet individual models (70 models)

# Extract unique identifiers from the training data
# This determines the total number of time series to model
unique_stores = time_series_data_train['store'].unique()      # 7 unique stores
unique_products = time_series_data_train['product'].unique()  # 10 unique products
# Total combinations: 7 × 10 = 70 individual time series

print(f"Dataset Overview:")
print(f"• Number of stores: {len(unique_stores)}")
print(f"• Number of products: {len(unique_products)}")
print(f"• Total store-product combinations: {len(unique_stores) * len(unique_products)}")

# =============================================================================
# APPROACH 1: AUTOREGRESSIVE (AR) INDIVIDUAL MODELS
# =============================================================================
# AutoReg models use past values to predict future values
# Individual approach: One model per store-product combination (70 total models)
# This captures unique patterns for each specific store-product pair
# Trade-off: High granularity but computationally intensive

from statsmodels.tsa.ar_model import AutoReg

print("\n\n=== FITTING INDIVIDUAL AUTOREGRESSIVE MODELS ===")
print("Using AutoReg with 365 lags for each Store-Product combination...")
print("This approach models each time series independently to capture unique patterns.")

# Counter for progress tracking during model fitting
ar_count = 0
total_ar_combinations = len(unique_stores) * len(unique_products)

for store in unique_stores:
    for product in unique_products:
        ar_count += 1
        print(f"Processing {ar_count}/{total_ar_combinations}: AutoReg for Store {store}, Product {product}")
        
        # =================================================================
        # DATA PREPARATION FOR CURRENT STORE-PRODUCT COMBINATION
        # =================================================================
        
        # Filter training data for the current store-product combination
        train_dataframe = time_series_data_train.loc[
            (time_series_data_train['store'] == store) & 
            (time_series_data_train['product'] == product)
        ].copy()
        
        # Clean and prepare training data structure
        train_dataframe.drop(columns=['store', 'product'], inplace=True)
        train_dataframe['Date'] = pd.to_datetime(train_dataframe['Date'], format='%Y-%m-%d')
        
        # Create DatetimeIndex with explicit daily frequency for time series modeling
        # This ensures consistent temporal structure required by AutoReg
        train_dataframe.set_index('Date', inplace=True)
        train_dataframe.index = pd.DatetimeIndex(train_dataframe.index, freq='D')

        # Prepare test data with identical structure for evaluation
        test_dataframe = time_series_data_test.loc[
            (time_series_data_test['store'] == store) & 
            (time_series_data_test['product'] == product)
        ].copy()
        
        # Clean and structure test data
        test_dataframe.drop(columns=['store', 'product'], inplace=True)
        test_dataframe['Date'] = pd.to_datetime(test_dataframe['Date'], format='%Y-%m-%d')
        
        # Create DatetimeIndex with explicit daily frequency
        test_dataframe.set_index('Date', inplace=True)
        test_dataframe.index = pd.DatetimeIndex(test_dataframe.index, freq='D')

        # =================================================================
        # ADVANCED DIFFERENCING STRATEGY FOR STATIONARITY
        # =================================================================
        
        # Apply hierarchical differencing to handle both seasonality and trend
        # Step 1: Seasonal differencing (365-day lag) - removes yearly patterns
        # This handles seasonal non-stationarity by comparing same day across years
        train_dataframe['seasonal_diff'] = (train_dataframe['number_sold'] - 
                                           train_dataframe['number_sold'].shift(365))
        train_dataframe.dropna(inplace=True)  # Remove NaN values from lag operation
        
        # Step 2: First-order differencing - removes remaining trend
        # This handles trend non-stationarity by focusing on period-to-period changes
        train_dataframe['diff'] = (train_dataframe['seasonal_diff'] - 
                                  train_dataframe['seasonal_diff'].shift(1))
        train_dataframe.dropna(inplace=True)  # Remove NaN values from differencing
        
        # Log differencing strategy for transparency
        print(f"  • Applied seasonal differencing (365-day lag) for yearly pattern removal")
        print(f"  • Applied first-order differencing for trend removal")
        print(f"  • Remaining data points after differencing: {len(train_dataframe)}")
        
        # Strategic note: This differencing approach is particularly effective for retail data
        # where yearly seasonality (holidays, seasons) and trends are prominent

        # =================================================================
        # AUTOREGRESSIVE MODEL FITTING AND FORECASTING
        # =================================================================
        
        # Fit AutoReg model with 365 lags to capture yearly dependencies
        # Using 365 lags allows the model to learn from the same day in previous years
        # This is crucial for retail data with strong yearly seasonal patterns
        ar_model = AutoReg(train_dataframe['number_sold'], lags=365)
        ar_fitted = ar_model.fit()
        
        # Generate out-of-sample forecasts for the test period
        # Forecasting 365 steps ahead to cover the full test year (2019)
        forecast = ar_fitted.predict(start=len(train_dataframe), 
                                   end=len(train_dataframe) + 364)
        
        # =================================================================
        # PERFORMANCE EVALUATION AND METRICS CALCULATION
        # =================================================================
        
        # Calculate Root Mean Square Error (RMSE) - measures prediction accuracy
        # Lower RMSE indicates better model performance
        rmse_value = np.sqrt(((test_dataframe['number_sold'] - forecast) ** 2).mean())
        
        # Calculate Mean Absolute Percentage Error (MAPE) - measures relative accuracy
        # MAPE provides scale-independent performance assessment
        mape_value = mean_absolute_percentage_error(test_dataframe['number_sold'], forecast)
        
        # Store comprehensive results for analysis and comparison
        results_list_AR.append({
            'store': store, 
            'product': product, 
            'RMSE': rmse_value, 
            'MAPE': mape_value, 
            'MAPE_Percentage': mape_value * 100,
            'model_type': 'individual_autoregressive',
            'data_points_used': len(train_dataframe),
            'forecast_horizon': len(test_dataframe)
        })

# =============================================================================
# AUTOREGRESSIVE RESULTS COMPILATION AND ANALYSIS
# =============================================================================

# Compile AutoReg results into a structured DataFrame for analysis
results_df_AR = pd.DataFrame(results_list_AR)

# Sort results by store and product for systematic analysis
results_df_AR.sort_values(by=['store', 'product'], inplace=True)
results_df_AR.reset_index(drop=True, inplace=True)

# Display comprehensive results summary
print("\n\n=== AUTOREGRESSIVE MODEL RESULTS ===")
print("Performance metrics comparing Test Data vs Forecasted Data (2019 full year)")
print(f"Successfully fitted {len(results_df_AR)} AutoReg models")
print("\nDetailed Results by Store-Product combination:")
print(results_df_AR)

# =============================================================================
# AUTOREGRESSIVE PERFORMANCE VISUALIZATION
# =============================================================================

# Create comprehensive visualization for AutoReg model performance
plt.figure(figsize=(20, 8))

# Plot average RMSE by store to identify store-level performance patterns
results_df_AR.groupby('store')['RMSE'].mean().plot(kind='bar', color='skyblue')
plt.title('Average RMSE for AutoReg Individual Models by Store')
plt.xlabel('Store ID')
plt.ylabel('Average RMSE')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

# Add value labels on bars for precise reading
ax = plt.gca()
for i, v in enumerate(results_df_AR.groupby('store')['RMSE'].mean().values):
    ax.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Display performance summary statistics
print(f"\n=== AUTOREGRESSIVE PERFORMANCE SUMMARY ===")
print(f"Average RMSE across all models: {results_df_AR['RMSE'].mean():.2f} (±{results_df_AR['RMSE'].std():.2f})")
print(f"Average MAPE: {results_df_AR['MAPE_Percentage'].mean():.2f}%")
print(f"Best performing model RMSE: {results_df_AR['RMSE'].min():.2f}")
print(f"Worst performing model RMSE: {results_df_AR['RMSE'].max():.2f}")
print(f"Range of performance: {results_df_AR['RMSE'].max() - results_df_AR['RMSE'].min():.2f}")

# Identify top and bottom performers for insights
best_model = results_df_AR.loc[results_df_AR['RMSE'].idxmin()]
worst_model = results_df_AR.loc[results_df_AR['RMSE'].idxmax()]

print(f"\n📈 Best performing combination: Store {best_model['store']}, Product {best_model['product']} (RMSE: {best_model['RMSE']:.2f})")
print(f"📉 Worst performing combination: Store {worst_model['store']}, Product {worst_model['product']} (RMSE: {worst_model['RMSE']:.2f})")

print("\n" + "="*60)


# =============================================================================
# INDIVIDUAL SARIMA MODELING - COMPREHENSIVE TIME SERIES APPROACH
# =============================================================================
# Approach 2: SARIMA (Seasonal AutoRegressive Integrated Moving Average) models
# Individual store-product models - same granularity as AutoReg approach for comparison
# SARIMA handles both trend (ARIMA) and seasonality components explicitly

from statsmodels.tsa.statespace.sarimax import SARIMAX

print("\n\n=== FITTING INDIVIDUAL SARIMA MODELS ===")
print("Using SARIMA(1,1,1)(1,1,1,365) for each Store-Product combination...")

# Counter for progress tracking
sarima_count = 0
total_sarima_combinations = len(unique_stores) * len(unique_products)

for store in unique_stores:
    for product in unique_products:
        sarima_count += 1
        print(f"Processing {sarima_count}/{total_sarima_combinations}: SARIMA for Store {store}, Product {product}")
        
        # Filter data for the current store-product combination from ORIGINAL data
        train_dataframe = time_series_data_train.loc[(time_series_data_train['store'] == store) & (time_series_data_train['product'] == product)].copy()
        train_dataframe.drop(columns=['store', 'product'], inplace=True)
        train_dataframe['Date'] = pd.to_datetime(train_dataframe['Date'], format='%Y-%m-%d')
        
        # Create DatetimeIndex with explicit daily frequency for SARIMA requirements
        train_dataframe.set_index('Date', inplace=True)
        train_dataframe.index = pd.DatetimeIndex(train_dataframe.index, freq='D')

        # Prepare test data with same structure
        test_dataframe = time_series_data_test.loc[(time_series_data_test['store'] == store) & (time_series_data_test['product'] == product)].copy()
        test_dataframe.drop(columns=['store', 'product'], inplace=True)
        test_dataframe['Date'] = pd.to_datetime(test_dataframe['Date'], format='%Y-%m-%d')
        
        # Create DatetimeIndex with explicit daily frequency
        test_dataframe.set_index('Date', inplace=True)
        test_dataframe.index = pd.DatetimeIndex(test_dataframe.index, freq='D')

        try:
            # Fit SARIMA model with yearly seasonality (365 days)
            # Model specification: SARIMA(1,1,1)(1,1,1,365)
            # Non-seasonal: AR(1) + I(1) + MA(1) - handles short-term dependencies and trend
            # Seasonal: AR(1) + I(1) + MA(1) with 365-day period - handles yearly patterns
            SARIMA_model = SARIMAX(train_dataframe['number_sold'], 
                                 order=(1, 1, 1),                    # (p,d,q) non-seasonal components
                                 seasonal_order=(1, 1, 1, 365))      # (P,D,Q,s) seasonal components
            SARIMA_results = SARIMA_model.fit()
            
            # Generate forecasts for the test period
            # Using out-of-sample prediction for true forecasting evaluation
            forecast = SARIMA_results.predict(start=len(train_dataframe), 
                                            end=len(train_dataframe) + len(test_dataframe) - 1)
            
            # Calculate performance metrics
            rmse_value = np.sqrt(((test_dataframe['number_sold'] - forecast) ** 2).mean())
            mape_value = mean_absolute_percentage_error(test_dataframe['number_sold'], forecast)
            
            # Store results with comprehensive metadata
            results_list_SARIMA_individual.append({
                'store': store, 
                'product': product, 
                'RMSE': rmse_value, 
                'MAPE': mape_value, 
                'MAPE_Percentage': mape_value * 100,
                'model_type': 'individual_sarima',
                'aic': SARIMA_results.aic,
                'bic': SARIMA_results.bic
            })
            
        except Exception as e:
            print(f"  ✗ SARIMA failed for Store {store}, Product {product}: {str(e)[:50]}...")
            # Record failed models with high error values for analysis
            # This helps identify problematic store-product combinations
            results_list_SARIMA_individual.append({
                'store': store, 
                'product': product, 
                'RMSE': 9999, 
                'MAPE': 9999, 
                'MAPE_Percentage': 999900,
                'model_type': 'individual_sarima_failed',
                'aic': np.nan,
                'bic': np.nan
            })

# =============================================================================
# INDIVIDUAL SARIMA RESULTS COMPILATION AND ANALYSIS
# =============================================================================

# Compile SARIMA results into a structured DataFrame
results_df_SARIMA = pd.DataFrame(results_list_SARIMA_individual)
results_df_SARIMA.sort_values(by=['store', 'product'], inplace=True)
results_df_SARIMA.reset_index(drop=True, inplace=True)

print("\n\n=== INDIVIDUAL SARIMA MODEL RESULTS ===")
successful_sarima = len(results_df_SARIMA[results_df_SARIMA['RMSE'] < 9999])
failed_sarima = len(results_df_SARIMA[results_df_SARIMA['RMSE'] >= 9999])

print(f"✓ Successfully fitted {successful_sarima} SARIMA models")
print(f"✗ Failed to fit {failed_sarima} SARIMA models")
print(f"Success rate: {(successful_sarima/total_sarima_combinations)*100:.1f}%")
print("\nDetailed Results:")
print(results_df_SARIMA)

# =============================================================================
# INDIVIDUAL SARIMA PERFORMANCE VISUALIZATION
# =============================================================================

# Create comprehensive visualization for SARIMA model performance
plt.figure(figsize=(20, 8))

# Filter out failed models (those with RMSE = 9999) for meaningful visualization
valid_sarima_results = results_df_SARIMA[results_df_SARIMA['RMSE'] < 9999]

if len(valid_sarima_results) > 0:
    # Plot average RMSE by store for valid SARIMA models
    valid_sarima_results.groupby('store')['RMSE'].mean().plot(kind='bar', color='lightcoral')
    plt.title('Average RMSE for Individual SARIMA Models by Store')
    plt.xlabel('Store')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Display performance summary statistics
    print(f"\n=== INDIVIDUAL SARIMA PERFORMANCE SUMMARY ===")
    print(f"Average RMSE: {valid_sarima_results['RMSE'].mean():.2f} (±{valid_sarima_results['RMSE'].std():.2f})")
    print(f"Average MAPE: {valid_sarima_results['MAPE_Percentage'].mean():.2f}%")
    print(f"Best RMSE: {valid_sarima_results['RMSE'].min():.2f}")
    print(f"Best MAPE: {valid_sarima_results['MAPE_Percentage'].min():.2f}%")
    print(f"Average AIC: {valid_sarima_results['aic'].mean():.2f}")
    print(f"Average BIC: {valid_sarima_results['bic'].mean():.2f}")
else:
    print("⚠️  No valid Individual SARIMA results to plot - all models failed")
    print("This suggests data issues or inappropriate model parameters for this dataset")

# =============================================================================
# CLUSTERED MODELING APPROACH - SARIMA BY PRODUCT
# =============================================================================
# Approach 3A: Aggregate all stores for each product and fit one SARIMA model per product
# This reduces the number of models from 70 (7×10) to just 10 models
# Assumes that products have similar seasonal patterns across different stores

print("\n\nFitting SARIMA models by Product (All stores combined per product)")

for product in unique_products:
    print(f"Fitting SARIMA for Product {product} (all stores)")
    
    # Aggregate sales across all stores for this specific product
    train_product = time_series_data_train[time_series_data_train['product'] == product].groupby('Date')['number_sold'].sum().reset_index()
    train_product['Date'] = pd.to_datetime(train_product['Date'])
    train_product.set_index('Date', inplace=True)
    train_product.index.freq = 'D'
    
    # Same aggregation for test data
    test_product = time_series_data_test[time_series_data_test['product'] == product].groupby('Date')['number_sold'].sum().reset_index()
    test_product['Date'] = pd.to_datetime(test_product['Date'])
    test_product.set_index('Date', inplace=True)
    test_product.index.freq = 'D'
    
    try:
        # Fit SARIMA model with yearly seasonality (1,1,1)(1,1,1,365)
        # This configuration handles both trend and seasonal components
        SARIMA_model = SARIMAX(train_product['number_sold'], 
                             order=(1, 1, 1),          # Non-seasonal: AR(1), I(1), MA(1)
                             seasonal_order=(1, 1, 1, 365))  # Seasonal: AR(1), I(1), MA(1) with 365-day period
        SARIMA_results = SARIMA_model.fit()
        
        # Generate forecasts for the test period
        forecast = SARIMA_results.predict(start=len(train_product), end=len(train_product) + len(test_product) - 1)
        
        # Calculate performance metrics
        rmse_value = np.sqrt(((test_product['number_sold'] - forecast) ** 2).mean())
        mape_value = mean_absolute_percentage_error(test_product['number_sold'], forecast)
        
        # Store results with metadata
        results_list_SARIMA_cluster.append({
            'store': 'All_Stores', 
            'product': product, 
            'RMSE': rmse_value, 
            'MAPE': mape_value, 
            'MAPE_Percentage': mape_value * 100,
            'approach': 'by_product'
        })
        
    except Exception as e:
        print(f"  ✗ SARIMA failed for Product {product}: {str(e)[:50]}...")
        # Record failed models with high error values for analysis
        results_list_SARIMA_cluster.append({
            'store': 'All_Stores', 
            'product': product, 
            'RMSE': 9999, 
            'MAPE': 9999, 
            'MAPE_Percentage': 999900,
            'approach': 'by_product'
        })

# =============================================================================
# CLUSTERED MODELING APPROACH - SARIMA BY STORE
# =============================================================================
# Approach 3B: Aggregate all products for each store and fit one SARIMA model per store
# This reduces the number of models from 70 (7×10) to just 7 models
# Assumes that stores have similar operational patterns across different products

print("\n\nFitting SARIMA models by Store (All products combined per store)")

for store in unique_stores:
    print(f"Fitting SARIMA for Store {store} (all products)")
    
    # Aggregate sales across all products for this specific store
    train_store = time_series_data_train[time_series_data_train['store'] == store].groupby('Date')['number_sold'].sum().reset_index()
    train_store['Date'] = pd.to_datetime(train_store['Date'])
    train_store.set_index('Date', inplace=True)
    train_store.index.freq = 'D'
    
    # Same aggregation for test data
    test_store = time_series_data_test[time_series_data_test['store'] == store].groupby('Date')['number_sold'].sum().reset_index()
    test_store['Date'] = pd.to_datetime(test_store['Date'])
    test_store.set_index('Date', inplace=True)
    test_store.index.freq = 'D'
    
    try:
        # Fit SARIMA model with yearly seasonality (same parameters as product-level models)
        SARIMA_model = SARIMAX(train_store['number_sold'], 
                             order=(1, 1, 1),          # Non-seasonal components
                             seasonal_order=(1, 1, 1, 365))  # Seasonal components with yearly period
        SARIMA_results = SARIMA_model.fit()
        
        # Generate forecasts for the test period
        forecast = SARIMA_results.predict(start=len(train_store), end=len(train_store) + len(test_store) - 1)
        
        # Calculate performance metrics
        rmse_value = np.sqrt(((test_store['number_sold'] - forecast) ** 2).mean())
        mape_value = mean_absolute_percentage_error(test_store['number_sold'], forecast)

        # Store results with metadata
        results_list_SARIMA_cluster.append({
            'store': store,
            'product': 'All_Products',
            'RMSE': rmse_value,
            'MAPE': mape_value,
            'MAPE_Percentage': mape_value * 100,
            'approach': 'by_store'
        })
        
    except Exception as e:
        print(f"  ✗ SARIMA failed for Store {store}: {str(e)[:50]}...")
        # Record failed models with high error values for analysis
        results_list_SARIMA_cluster.append({
            'store': store,
            'product': 'All_Products',
            'RMSE': 9999,
            'MAPE': 9999,
            'MAPE_Percentage': 999900,
            'approach': 'by_store'
        })

# =============================================================================
# CLUSTERED SARIMA RESULTS ANALYSIS
# =============================================================================

# Compile and display results from both clustering approaches
results_df_SARIMA_agg = pd.DataFrame(results_list_SARIMA_cluster)
results_df_SARIMA_agg.sort_values(by=['store', 'product'], inplace=True)
results_df_SARIMA_agg.reset_index(drop=True, inplace=True)

print("\n\n=== SARIMA CLUSTERED MODELING RESULTS ===")
print("Product-level models (10 models) + Store-level models (7 models) = 17 total models")
print("Compare this to 70 individual models in the traditional approach")
print(results_df_SARIMA_agg)

# =============================================================================
# PERFORMANCE VISUALIZATION - CLUSTERED SARIMA MODELS
# =============================================================================

# Create visualization for SARIMA clustered model performance
plt.figure(figsize=(20, 8))

# Filter out failed models (those with RMSE = 9999) for meaningful visualization
valid_results_agg = results_df_SARIMA_agg[results_df_SARIMA_agg['RMSE'] < 9999]

if len(valid_results_agg) > 0:
    # Plot average RMSE by store for valid models
    valid_results_agg.groupby('store')['RMSE'].mean().plot(kind='bar', color='lightcoral')
    plt.title('Average RMSE for SARIMA Clustered Models by Store')
    plt.xlabel('Store/Grouping')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("⚠️  No valid SARIMA Clustered results to plot - all models failed")

# =============================================================================
# PROPHET MODELING APPROACH - INDIVIDUAL STORE-PRODUCT MODELS
# =============================================================================
# Approach 4: Use Facebook Prophet for time series forecasting
# Prophet is designed to handle seasonality, trends, and holidays automatically
# It's particularly good at handling missing data and outliers

from prophet import Prophet

print("\n\n=== FITTING PROPHET MODELS ===")
print("Using Prophet for individual Store-Product combinations...")

for store in unique_stores:
    for product in unique_products:
        print(f"Fitting PROPHET for Store {store}, Product {product}")
        
        # Filter data for the current store-product combination
        train_dataframe = time_series_data_train.loc[(time_series_data_train['store'] == store) & (time_series_data_train['product'] == product)].copy()
        train_dataframe.drop(columns=['store', 'product'], inplace=True)
        train_dataframe['Date'] = pd.to_datetime(train_dataframe['Date'], format='%Y-%m-%d')

        # Prophet requires specific column names: 'ds' for dates, 'y' for values
        train_prophet = train_dataframe.rename(columns={'Date': 'ds', 'number_sold': 'y'})
        
        # Prepare test data for evaluation
        test_dataframe = time_series_data_test.loc[(time_series_data_test['store'] == store) & (time_series_data_test['product'] == product)].copy()
        test_dataframe.drop(columns=['store', 'product'], inplace=True)
        test_dataframe['Date'] = pd.to_datetime(test_dataframe['Date'], format='%Y-%m-%d')
        test_prophet = test_dataframe.rename(columns={'Date': 'ds', 'number_sold': 'y'})
        
        try:
            # Initialize Prophet model with yearly seasonality
            # Prophet automatically detects weekly patterns and can handle yearly seasonality
            prophet_model = Prophet(
                yearly_seasonality=True,    # Enable yearly seasonal patterns
                weekly_seasonality=True,    # Enable weekly seasonal patterns  
                daily_seasonality=False,    # Disable daily seasonality (not relevant for this data)
                interval_width=0.95         # 95% confidence intervals
            )
            
            # Fit the model on training data
            prophet_model.fit(train_prophet)
            
            # Create future dataframe for forecasting
            # Prophet needs future dates to make predictions
            future_dates = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='D')
            
            # Generate forecasts
            forecast_prophet = prophet_model.predict(future_dates)
            
            # Extract forecasts for the test period only
            test_forecast = forecast_prophet.tail(len(test_prophet))['yhat'].values
            
            # Calculate performance metrics
            rmse_value = np.sqrt(((test_prophet['y'] - test_forecast) ** 2).mean())
            mape_value = mean_absolute_percentage_error(test_prophet['y'], test_forecast)
            
            # Store results
            results_list_prophet.append({
                'store': store, 
                'product': product, 
                'RMSE': rmse_value, 
                'MAPE': mape_value, 
                'MAPE_Percentage': mape_value * 100
            })
            
        except Exception as e:
            print(f"  ✗ PROPHET failed for Store {store}, Product {product}: {str(e)[:50]}...")
            # Record failed models for analysis
            results_list_prophet.append({
                'store': store, 
                'product': product, 
                'RMSE': 9999, 
                'MAPE': 9999, 
                'MAPE_Percentage': 999900
            })

# =============================================================================
# PROPHET RESULTS COMPILATION AND ANALYSIS
# =============================================================================

# Compile Prophet results into a DataFrame for analysis
results_df_prophet = pd.DataFrame(results_list_prophet)
results_df_prophet.sort_values(by=['store', 'product'], inplace=True)
results_df_prophet.reset_index(drop=True, inplace=True)

print("\n\n=== PROPHET MODEL RESULTS ===")
print(f"Successfully fitted {len(results_df_prophet[results_df_prophet['RMSE'] < 9999])} Prophet models")
print(f"Failed to fit {len(results_df_prophet[results_df_prophet['RMSE'] >= 9999])} Prophet models")
print(results_df_prophet)

# =============================================================================
# PROPHET PERFORMANCE VISUALIZATION
# =============================================================================

# Create visualization for Prophet model performance
plt.figure(figsize=(20, 8))

# Filter out failed models (those with RMSE = 9999) for meaningful visualization
valid_results_prophet = results_df_prophet[results_df_prophet['RMSE'] < 9999]

if len(valid_results_prophet) > 0:
    # Plot average RMSE by store for valid models
    valid_results_prophet.groupby('store')['RMSE'].mean().plot(kind='bar', color='lightgreen')
    plt.title('Average RMSE for Prophet Models by Store')
    plt.xlabel('Store')
    plt.ylabel('Average RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Summary statistics for Prophet models
    print(f"\n=== PROPHET PERFORMANCE SUMMARY ===")
    print(f"Average RMSE: {valid_results_prophet['RMSE'].mean():.2f} (±{valid_results_prophet['RMSE'].std():.2f})")
    print(f"Average MAPE: {valid_results_prophet['MAPE'].mean():.4f} ({valid_results_prophet['MAPE_Percentage'].mean():.2f}%)")
    print(f"Best RMSE: {valid_results_prophet['RMSE'].min():.2f}")
    print(f"Best MAPE: {valid_results_prophet['MAPE'].min():.4f} ({valid_results_prophet['MAPE_Percentage'].min():.2f}%)")
else:
    print("⚠️  No valid Prophet results to plot - all models failed")

# =============================================================================
# COMPREHENSIVE MODEL COMPARISON
# =============================================================================

print("\n\n" + "="*80)
print("                    FINAL MODEL COMPARISON SUMMARY")
print("="*80)

# Create a comprehensive comparison of all modeling approaches
print("\n📊 MODEL COMPLEXITY COMPARISON:")
print("• AutoReg Individual Models: 70 models (7 stores × 10 products)")
print("• SARIMA Clustered Models: 17 models (10 by product + 7 by store)")  
print(f"• Prophet Individual Models: {len(results_df_prophet)} models (attempted)")

print("\n📈 PERFORMANCE METRICS:")
print("AutoReg (Individual):")
print(f"  - Average RMSE: {results_df_AR['RMSE'].mean():.2f}")
print(f"  - Average MAPE: {results_df_AR['MAPE_Percentage'].mean():.2f}%")

if len(valid_results_agg) > 0:
    print("SARIMA (Clustered):")
    print(f"  - Average RMSE: {valid_results_agg['RMSE'].mean():.2f}")
    print(f"  - Average MAPE: {valid_results_agg['MAPE_Percentage'].mean():.2f}%")

if len(valid_results_prophet) > 0:
    print("Prophet (Individual):")
    print(f"  - Average RMSE: {valid_results_prophet['RMSE'].mean():.2f}")
    print(f"  - Average MAPE: {valid_results_prophet['MAPE_Percentage'].mean():.2f}%")

print("\n🎯 MODELING INSIGHTS:")
print("• Individual models capture store-product specific patterns")
print("• Clustered models reduce complexity while maintaining reasonable accuracy")
print("• Prophet automatically handles seasonality and missing data")
print("• Choice depends on accuracy requirements vs computational resources")

print("\n" + "="*80)
