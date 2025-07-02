# Real Estate Price Prediction

This project analyzes a real estate dataset, builds a predictive model for house prices, and provides insights into the factors influencing prices.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Data Loading and Exploration](#data-loading-and-exploration)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Data Preprocessing](#data-preprocessing)
5.  [Model Training and Evaluation](#model-training-and-evaluation)
6.  [Feature Importance](#feature-importance)
7.  [User Input and Prediction](#user-input-and-prediction)
8.  [Summary of Findings](#summary-of-findings)

## 1. Introduction

This notebook demonstrates a process for predicting real estate prices based on various features such as transaction date, house age, distance to MRT station, number of convenience stores, latitude, and longitude. The process involves data loading, exploration, visualization, preprocessing, model training, evaluation, and prediction.

## 2. Data Loading and Exploration

The dataset `Real_Estate.csv` is loaded into a pandas DataFrame. Basic exploration is performed to view the head and tail of the DataFrame, check its information (data types, non-null counts), and identify missing values.

-   Loading data using `pd.read_csv()`.
-   Viewing data using `df.head()`, `df.tail()`, and `display(df)`.
-   Checking data information using `df.info()`.
-   Checking for null values using `df.isnull().sum()`.

## 3. Exploratory Data Analysis (EDA)

Various visualizations are generated to understand the distribution of individual features and the relationships between features.

-   **Basic Visualizations:** Histograms and box plots for numerical features (`House age`, `Distance to the nearest MRT station`, `Number of convenience stores`) to visualize distributions and outliers. Scatter plots to visualize the relationship between 'House price of unit area' and other numerical features.
-   **Correlation Analysis:** A heatmap of the correlation matrix to understand the linear relationships between numerical features.
-   **Advanced Visualizations (Optional):** Pair plots and joint plots for a comprehensive view of pairwise relationships and individual distributions.

## 4. Data Preprocessing

The data is preprocessed to prepare it for machine learning models.

-   The 'Transaction date' is converted to a numerical format (Unix timestamp).
-   Features (X) and the target variable (y - 'House price of unit area') are separated.
-   Numerical features are scaled using `StandardScaler` to standardize their ranges.
-   The data is split into training and testing sets using `train_test_split`.
-   *(Note: An engineered 'House Price' feature was created for VIF analysis, but it's important to exclude target variables or features derived from the target during actual model training and prediction to avoid data leakage.)*

## 5. Model Training and Evaluation

Several regression models are trained on the preprocessed training data and evaluated on the testing data.

-   Models trained: `RandomForestRegressor`, `GradientBoostingRegressor`, and `SVR`.
-   Models are trained using the `fit()` method.
-   Predictions are made on the test set using the `predict()` method.
-   Evaluation metrics calculated: Root Mean Squared Error (RMSE) and R2 Score.

## 6. Feature Importance

The importance of each feature in the trained models (specifically Random Forest and Gradient Boosting) is analyzed.

-   Feature importance scores are obtained from the trained models.
-   Feature importance is visualized using bar plots (though the code for visualizing only the Linear Regression feature importance was shown, the concept applies to other models).
-   *(Note: The engineered 'House Price' feature was incorrectly included in the feature importance analysis in the notebook. This should be excluded in a real-world scenario.)*

## 7. User Input and Prediction

Code is provided to take user input for the features, preprocess it using the same steps as the training data, and make a prediction using the selected optimal model.

-   A dictionary is used to store sample user input.
-   User input is converted to a pandas DataFrame.
-   'Transaction date' is converted to the numerical format.
-   Columns are reordered and a placeholder 'House Price' column is added to match the training data structure.
-   Numerical features in the user input DataFrame are scaled.
-   The optimal model (Gradient Boosting Regressor in this case, based on evaluation metrics) is used to predict the house price.
-   The predicted house price is displayed.

## 8. Summary of Findings

-   Location-based features (`Distance to the nearest MRT station`, `Number of convenience stores`, `Latitude`, and `Longitude`) are the most influential factors in predicting house prices in this dataset.
-   Gradient Boosting and Random Forest models performed significantly better than the Support Vector Regressor.
-   The model evaluation visualizations (actual vs. predicted prices, residuals plots) suggest that the chosen model provides a reasonable fit to the data.
-   *(Important Note: Data leakage occurred by including an engineered feature derived from the target variable in the training data and feature importance analysis. This should be corrected in a real application.)*

This README provides a high-level overview of the steps taken in the notebook. Refer to the notebook itself for the detailed code and outputs.
