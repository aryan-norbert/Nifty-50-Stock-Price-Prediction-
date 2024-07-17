# Nifty 50 Stock Price Prediction

This project aims to predict the stock prices of Nifty 50 companies using machine learning models. The project includes data collection, preprocessing, exploratory data analysis, model training, evaluation, and visualization of predictions.

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter Notebooks for data analysis and model training.
- `src/`: Python scripts for data preprocessing, model training, evaluation, and visualization.
- `models/`: Directory to store the trained models.
- `requirements.txt`: List of dependencies.
- `README.md`: Project overview and instructions.

## Setup Instructions

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/nifty50-stock-prediction.git
    cd nifty50-stock-prediction
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download the dataset:**
    - Ensure you have the dataset in the `data/` directory or specify the data source in the preprocessing script.

5. **Run the data preprocessing script:**

    ```sh
    python src/data_preprocessing.py
    ```

6. **Train the models:**

    ```sh
    python src/model_training.py
    ```

7. **Evaluate the models:**

    ```sh
    python src/model_evaluation.py
    ```

## Key Features

1. **Data Collection and Preprocessing**:
    - Loading and cleaning the dataset.
    - Handling missing values and scaling numerical features.
    - Feature engineering.

2. **Exploratory Data Analysis (EDA)**:
    - Visualizing stock price trends and patterns.
    - Analyzing feature correlations.

3. **Model Training**:
    - Implementing machine learning models such as Linear Regression, Random Forest, and LSTM.
    - Splitting the dataset into training and testing sets.
    - Hyperparameter tuning using GridSearchCV.

4. **Model Evaluation**:
    - Evaluating model performance using metrics like RMSE and MAE.
    - Visualizing predicted vs. actual stock prices.

5. **Visualization and Insights**:
    - Plotting stock price trends and model predictions.
    - Analyzing feature importances and correlations.

## Usage

1. **Run the preprocessing script** to clean and prepare the data:

    ```sh
    python src/data_preprocessing.py
    ```

2. **Train and evaluate the models**:

    ```sh
    python src/model_training.py
    python src/model_evaluation.py
    ```

3. **Visualize the results**:
    - Stock price trends.
    - Predicted vs. actual prices.

## License

This project is licensed under the MIT License.

