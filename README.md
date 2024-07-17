# Stock Predictions Project

This project focuses on predicting stock prices using machine learning models. It employs two main algorithms: Linear Regression and LSTM (Long Short-Term Memory).

### Methodology:
1. **Data Preprocessing**: The project begins with data preprocessing using Pandas and NumPy libraries in Python. Historical stock data, typically in CSV format, is cleaned by handling missing values, scaling numerical features, and splitting the data into training and testing sets.

2. **Model Implementation**:
   - **Linear Regression**: This model is chosen for its simplicity and interpretability in fitting a linear relationship between input features (e.g., historical stock prices, trading volume) and the target variable (future stock price).
   - **LSTM**: Long Short-Term Memory is employed for its ability to capture long-term dependencies in time series data. TensorFlow is used to implement the LSTM model, which processes sequential data for better forecasting accuracy.

3. **Evaluation**: The performance of these models is evaluated using metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) to measure prediction accuracy.

### Libraries Used:
- Python 3.x
- Pandas: Data manipulation and preprocessing.
- NumPy: Numerical operations and array manipulation.
- Scikit-learn: Machine learning algorithms, including Linear Regression.
- TensorFlow: Deep learning framework, used for LSTM implementation.

