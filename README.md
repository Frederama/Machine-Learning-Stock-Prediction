# Machine Learning Stock Prediction

## Objective
Predict 5-day forward log returns of Microsoft (MSFT) stock using supervised machine learning models and engineered financial features. The objective is to build a robust, end-to-end ML pipeline that evaluates both predictive accuracy and real-world trading viability while guarding against data snooping.

## Dataset
- **Source:** 
  - Yahoo Finance – MSFT, GOOGL, IBM stock data
  - Federal Reserve Economic Data (FRED) – Market indices and FX rates
- **Time Range:** January 2010 – July 2025 (daily data, aggregated to 5-day returns)
- **Features:**
  - MSFT OHLCV data
  - Lagged MSFT log returns (multiple horizons)
  - Correlated assets (GOOGL, IBM)
  - Market indices (S&P 500, DJIA, VIX)
  - FX rates (DEXJPUS, DEXUSUK)
  - Custom technical indicators (momentum, volatility, RSI, ADX)
- **Target:** 5-day log return of MSFT closing price (continuous regression target)

## Approach
This project implements a full machine learning pipeline for financial time-series prediction:

- Performed exploratory data analysis (EDA) and feature distribution analysis
- Converted price data to log returns to stabilize variance and improve stationarity
- Engineered lagged returns, rolling statistics, and technical indicators
- Applied feature selection using SelectKBest (F-score)
- Used time-aware train/test split to prevent look-ahead bias
- Evaluated 12 regression models using cross-validation:
  - **Linear, LASSO, ElasticNet**  
  - **KNN, Decision Tree, SVR**  
  - **MLP Neural Network**  
  - **AdaBoost, Gradient Boosting**
  - **XGBoost, Random Forest, Extra Trees**
- Selected Random Forest Regressor for final modeling
- Tuned hyperparameters using Randomized Search with TimeSeriesSplit
- Generated trading signals from predicted returns and evaluated strategy performance
- Interpreted model behavior using SHAP values
- Validated robustness using White’s Reality Check and Monte Carlo permutation tests

## Results
- **Best Model:** Random Forest Regressor  
- **Statistical Performance (Test Set):** MSE: 0.000861 and MAE: 0.023
- **Financial Performance (Out-of-Sample):** Profit Factor: ~1.85, Sharpe Ratio: ~1.86 and CAGR: ~42.6%
- **Validation:**
  - White’s Reality Check: p = 0.0306 (reject null; performance not due to chance)
  - Monte Carlo Permutation Test: p = 0.003 (profitability highly unlikely by randomness)
The model generalizes well with no signs of overfitting and demonstrates statistically and financially meaningful predictive power.

## Tools Used
`Python` · `Pandas` · `Numpy` · `Scikit-learn` · `Matplotlib` · `SHAP` · `Statsmodels` · `Yahoo Finance` · `FRED` · `Jupyter Notebook`

## References
- Tatsat, H., de Jong, N., & Bacaj, J. Machine Learning for Finance (O’Reilly)
- White, H. (2000). A Reality Check for Data Snooping
- Masters, T. (2021). Testing and Tuning Market Trading Systems
- Yahoo Finance
- Federal Reserve Economic Data (FRED)
- Scikit-learn Documentation
- SHAP Documentation

---

*Check out the full code and notebook in this repository!*
