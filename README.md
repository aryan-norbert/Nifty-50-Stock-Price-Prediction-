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

