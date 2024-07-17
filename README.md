# Nifty 50 Stock Price Prediction

## Overview
This repository provides Python scripts for predicting Nifty 50 stock prices using historical data sourced from Yahoo Finance. The project focuses on preprocessing data, building predictive models, evaluating their performance, and visualizing results.

## Features
- **Data Preprocessing**: Handles missing values, computes moving averages (SMA), and calculates Relative Strength Index (RSI).
- **Feature Engineering**: Generates lagged features of closing prices to capture temporal dependencies.
- **Model Development**: Utilizes linear regression models from both `sklearn` and `statsmodels`.
- **Evaluation**: Assesses model performance using Mean Squared Error (MSE).
- **Visualization**: Includes scripts to plot actual versus predicted stock prices for visual inspection of model accuracy.

## Requirements
- Python 3.x
- Required libraries: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `statsmodels`

## Installation
1. Clone the repository:
https://github.com/aryan-norbert/Nifty-50-Stock-Price-Prediction-.git

2. Install dependencies:


## Usage
1. Run `preprocess_data.py` to fetch historical data, preprocess it, and save as CSV.
2. Execute `train_model.py` to train linear regression models and evaluate their performance.
3. Use `visualize_results.py` to generate plots comparing predicted and actual stock prices.

## Example
```bash
python preprocess_data.py
python train_model.py
python visualize_results.py


