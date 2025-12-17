# Credit-Spread-Forecasting-A-Study-of-How-Machine-Learning-Can-Enhance-Predictive-Accuracy
Credit Spread forecasting via Machine Learning (XGBoost, RF) and Deep Learning (TCN). The project integrates mixed-frequency macro data using MIDAS algorithms and employs Hidden Markov Models (HMM) for regime detection, significantly enhancing predictive accuracy and regime-conditioned risk analysis (VaR/ES).
# Credit Spread Forecasting: Machine Learning, MIDAS & Regime Switching

**Author:** Giorgio Perego  
**Academic Context:** Master's Thesis in Economics and Finance  
**Institution:** University of Milano-Bicocca

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Research-yellow)

## 📄 Abstract

This repository contains the source code for the empirical analysis conducted in the thesis **"Credit Spread Forecasting: A Study of How Machine Learning Can Enhance Predictive Accuracy."**

The project tackles the "Credit Spread Puzzle" by comparing traditional econometric approaches with advanced Machine Learning (Random Forest, XGBoost) and Deep Learning (Temporal Convolutional Networks) models. It specifically addresses the challenge of mixing data frequencies (Daily market data vs. Monthly/Quarterly Macro data) using **MIDAS (Mixed Data Sampling)** techniques and incorporates **Hidden Markov Models (HMM)** to account for shifting market regimes.

## 🚀 Key Features

* **MIDAS Transformation:** Custom implementation of "Reverse MIDAS" with Almon/Beta lag polynomials to seamlessly blend low-frequency macroeconomic indicators (GDP, PCE) with high-frequency financial data.
* **Regime Detection (HMM):** Utilization of Gaussian HMM to unsupervisedly classify market states (e.g., Stable vs. Volatile) and condition risk metrics.
* **Advanced Models:**
    * **TCN (Temporal Convolutional Network):** A fast, streaming implementation using `PyTorch` with dilated convolutions for capturing long-term dependencies.
    * **Ensemble Methods:** Early-stopping Random Forests and Bayesian-optimized XGBoost.
    * **Factor Analysis:** Dimensionality reduction to extract latent economic drivers from macro variables.
* **Risk Analysis:** Computation of Regime-Conditioned Value at Risk (VaR) and Expected Shortfall (ES).

## 🛠️ Code Architecture

The codebase is structured into two distinct pipelines to test different hypotheses:

### Pipeline 1: The ML & Deep Learning Approach
* **Focus:** Maximizing predictive accuracy and capturing non-linearities.
* **Components:**
    * Stationary Feature Engineering.
    * Regime feature extraction via HMM.
    * Model training: TCN, XGBoost, Random Forest.
    * **Output:** Comparative metrics (RMSE, MAE, R²) and Rolling VaR analysis.

### Pipeline 2: The Causal & Econometric Approach
* **Focus:** Explainability and structural validity.
* **Components:**
    * Strict Causal Split (Train/Test split occurs *before* transformations to prevent data leakage).
    * Factor Analysis (FA) to condense macro variables.
    * Log-Linear Regression wrapper.
    * Feature Importance analysis.

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/credit-spread-forecasting.git](https://github.com/your-username/credit-spread-forecasting.git)
    cd credit-spread-forecasting
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration & Data

### Data Sources
The code automatically fetches data from:
* **FRED (Federal Reserve Economic Data):** Via `pandas_datareader`.
* **Yahoo Finance:** Via `yfinance`.
* **Local Files:** TED Rate and Policy Uncertainty data (Excel/CSV).

### Setup Paths
**Important:** You must configure the local file paths in the `CommonConfig` class before running the script. Open `main.py` and modify this section:

```python
class CommonConfig:
    START_DATE = '2000-01-01'
    # Modify with your local paths
    YAHOO_SERIES = {
        'SPY_DIFF': '/path/to/your/file/Spy.xlsx'
    }
    PATHS = {
        'TED_RATE': '/path/to/your/file/tedspread2.xlsx',
        'POLICY': '/path/to/your/file/Categorical_EPU_Data2.csv'
    }
