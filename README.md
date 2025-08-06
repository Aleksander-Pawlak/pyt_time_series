# Time Series Forecasting: Multi-Store Retail Sales Analysis

A comprehensive time series forecasting project comparing multiple modeling approaches for retail sales prediction across store-product combinations.

## 📊 Project Overview

This project implements and compares **four different time series forecasting approaches** to predict daily sales across 7 stores and 10 products (70 unique combinations). The analysis covers exploratory data analysis, stationarity testing, and comprehensive model comparison using real retail sales data.

### 🎯 Key Objectives
- Compare individual vs. aggregated modeling strategies
- Evaluate traditional statistical models vs. modern forecasting tools
- Analyze performance trade-offs between model complexity and accuracy
- Provide actionable insights for retail forecasting applications

## 📁 Dataset Structure

```
📦 Data Files
├── train.csv    # Training data (2017-2018)
├── test.csv     # Test data (2019)
└── main.py      # Complete analysis pipeline
```

**Data Schema:**
- `Date`: Daily observations (YYYY-MM-DD format)
- `store`: Store identifier (0-6, 7 unique stores)
- `product`: Product identifier (0-9, 10 unique products) 
- `number_sold`: Daily sales quantity (target variable)

**Time Period:**
- **Training**: 2017-01-01 to 2018-12-31 (2 years)
- **Testing**: 2019-01-01 to 2019-12-31 (1 year)
- **Frequency**: Daily observations with 365-day seasonal patterns

## 🔬 Methodology & Analysis Pipeline

### 1. **Exploratory Data Analysis (EDA)**
- **Rolling Statistics**: Multi-timeframe trend analysis (7, 90, 180, 365 days)
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) tests
- **Autocorrelation Analysis**: ACF/PACF plots for pattern identification
- **Seasonal Decomposition**: Trend, seasonal, and residual component separation
- **Differencing Strategy**: Hierarchical approach for stationarity achievement

### 2. **Modeling Approaches**

#### 🔹 **Approach 1: Individual AutoRegressive (AR) Models**
- **Scope**: 70 individual models (one per store-product combination)
- **Model**: AutoReg with 365 lags for yearly seasonality capture
- **Preprocessing**: Advanced hierarchical differencing (seasonal + trend)
- **Advantages**: Maximum granularity, captures unique patterns
- **Trade-offs**: Computationally intensive, 70 models to maintain

#### 🔹 **Approach 2: Individual SARIMA Models** 
- **Scope**: 70 individual models with explicit seasonality handling
- **Model**: SARIMA(1,1,1)(1,1,1,365) - comprehensive ARIMA + seasonality
- **Features**: Built-in trend and seasonal differencing, MA components
- **Advantages**: Theoretically robust, handles complex patterns
- **Trade-offs**: Computationally expensive, potential convergence issues

#### 🔹 **Approach 3: Clustered SARIMA Models**
- **3A - By Product**: 10 models (aggregate all stores per product)
- **3B - By Store**: 7 models (aggregate all products per store)
- **Total**: 17 models vs. 70 individual models
- **Advantages**: Reduced complexity, faster training, easier maintenance
- **Assumptions**: Similar patterns within clusters

#### 🔹 **Approach 4: Prophet Individual Models**
- **Scope**: 70 individual models using Facebook Prophet
- **Features**: Automatic seasonality detection, holiday effects, missing data handling
- **Configuration**: Yearly + weekly seasonality, 95% confidence intervals
- **Advantages**: User-friendly, robust to missing data, interpretable
- **Use Case**: Business-friendly forecasting with automatic feature engineering

### 3. **Performance Evaluation**

**Metrics Used:**
- **RMSE (Root Mean Square Error)**: Absolute prediction accuracy
- **MAPE (Mean Absolute Percentage Error)**: Scale-independent relative accuracy
- **AIC/BIC**: Model selection criteria (where applicable)

**Evaluation Framework:**
- Out-of-sample forecasting on 2019 test data
- Store-level and overall performance aggregation
- Success rate tracking for model convergence
- Statistical significance testing

## 🛠️ Technical Implementation

### **Core Dependencies**
```python
pandas>=2.3.1          # Data manipulation and analysis
numpy>=2.3.2           # Numerical computations
matplotlib>=3.10.3     # Visualization
seaborn>=0.13.2        # Statistical visualization
statsmodels            # Time series modeling (AutoReg, SARIMA)
scikit-learn           # Evaluation metrics
prophet                # Facebook Prophet forecasting
```

### **Key Technical Features**
- **Frequency Management**: Explicit daily frequency handling with `asfreq()`
- **Missing Data**: Automatic gap filling for irregular time series
- **Memory Optimization**: Efficient data filtering and copying strategies
- **Error Handling**: Comprehensive exception management for model failures
- **Progress Tracking**: Real-time model fitting progress indicators

## 📈 Key Findings & Insights

### **Model Complexity vs. Performance**
```
Individual Models (70):     High accuracy, high complexity
Clustered Models (17):      Balanced accuracy, moderate complexity  
Prophet Models (70):        Automatic features, user-friendly
```

### **Business Applications**
- **High-accuracy needs**: Individual AutoReg/SARIMA models
- **Operational efficiency**: Clustered SARIMA models
- **Business stakeholders**: Prophet models with interpretable outputs
- **Resource constraints**: Clustered approaches for reduced maintenance

### **Performance Characteristics**
- **AutoReg**: Excellent for patterns with strong yearly dependencies
- **SARIMA**: Best theoretical foundation, handles complex seasonality
- **Clustered**: Good balance of accuracy and operational simplicity
- **Prophet**: Robust to outliers, automatic holiday detection

## 🚀 Usage Instructions

### **Running the Complete Analysis**
```bash
# Ensure data files are in the project directory
# train.csv and test.csv with proper schema

python main.py
```

### **Key Outputs**
1. **Exploratory Analysis**: Rolling statistics, stationarity tests, decomposition plots
2. **Model Performance**: RMSE/MAPE metrics for all approaches
3. **Comparative Visualizations**: Store-level performance comparisons
4. **Business Insights**: Best/worst performing combinations, success rates

### **Customization Options**
- **Seasonal Period**: Modify `period=365` for different seasonality
- **Model Parameters**: Adjust SARIMA orders `(p,d,q)(P,D,Q,s)`
- **Evaluation Metrics**: Add custom performance measures
- **Clustering Strategy**: Implement alternative aggregation approaches

## 📋 Project Structure

```
pyt_time_series/
├── main.py              # Complete analysis pipeline
├── train.csv            # Training dataset (2017-2018)
├── test.csv             # Test dataset (2019)
├── README.md            # Project documentation
└── outputs/             # Generated plots and results (created during run)
```

## 🔍 Advanced Features

### **Differencing Strategy**
```python
# Hierarchical differencing for stationarity
seasonal_diff = data - data.shift(365)    # Remove yearly patterns
trend_diff = seasonal_diff - seasonal_diff.shift(1)  # Remove trend
```

### **Model Selection Criteria**
- **AutoReg**: 365 lags for yearly dependency modeling
- **SARIMA**: (1,1,1)(1,1,1,365) for comprehensive pattern capture
- **Prophet**: Automatic parameter selection with business-friendly defaults

### **Error Analysis**
- Failed model tracking with high error flags (RMSE=9999)
- Success rate calculation and convergence diagnostics
- Store-product combination performance ranking

## 📊 Expected Results

The analysis provides:
1. **Performance Rankings**: Which modeling approach works best for your data
2. **Store Insights**: Which stores/products are easier/harder to forecast
3. **Complexity Analysis**: Computational vs. accuracy trade-offs
4. **Business Recommendations**: Optimal modeling strategy for different use cases

## 🤝 Contributing

This project serves as a comprehensive template for retail time series forecasting. Key areas for extension:
- Additional seasonal decomposition methods
- Ensemble modeling approaches
- External factor integration (holidays, promotions)
- Real-time forecasting pipeline implementation

## 📄 License

Open source project for educational and research purposes. Feel free to adapt for your own time series forecasting needs.

---

**Author**: Time Series Forecasting Analysis  
**Last Updated**: August 2025  
**Python Version**: 3.8+  
**Analysis Type**: Comparative Time Series Forecasting Study
