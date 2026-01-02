import pandas as pd
import numpy as np
# import Scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data_frame = pd.read_csv('./data/cars-data.csv')

# Data Separation as X and Y
# X = Feature Set (Training Variables/Data)
# Y = Target Variable (What we want to predict)

# Here, we want to predict the 'selling_price' of cars

# Apply logarithmic transformation to the target variable to normalize its scale
# Normalization helps in improving model performance and interpretability of error metrics
# Normalization vs Scaling:
# Normalization means adjusting values measured on different scales to a common scale,
# Scaling means adjusting the range of features or target variable to a standard scale 
# without distorting differences in the ranges of values
# This will reduce the scale of the target variable and make the MSE more interpretable.
y = data_frame["selling_price"].apply(lambda x: np.log1p(x))
# The feature set X is obtained by dropping the target variable column from the DataFrame
# X is uppercase because it represents a matrix (2D array)
X = data_frame.drop(columns=["selling_price"])

# Identify and encode categorical/object/string (non-numeric) columns
# step 1 : select columns with object data type
categorical_columns = X.select_dtypes(include=['object']).columns
if not categorical_columns.empty:
    # step 2 : apply one-hot encoding using pd.get_dummies
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the dataset into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# X_train: Training features
# X_test: Testing features
# y_train: Training target variable
# y_test: Testing target variable
# random_state=42 ensures reproducibility of the results
# 42 is just a commonly used arbitrary number for the random seed but can be any integer
# another common choice is 0, 1, or 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# Building Machine Learning Model - Linear Regression
# =========================================
lr = LinearRegression() # Create an instance of the Linear Regression model
lr.fit(X_train, y_train) # Train the model using the training data both features and target variable

# Now the model is trained and can be used to make predictions
# Evaluating the model's performance on the test set
lr_predict_y_train = lr.predict(X_train) # Make predictions using training data
lr_predict_y_test = lr.predict(X_test) # Make predictions using testing data

# Calculate evaluation metrics for training and testing sets
lr_train_mse = mean_squared_error(y_train, lr_predict_y_train) # Mean Squared Error for training set
lr_train_r2 = r2_score(y_train, lr_predict_y_train) # R-squared for training set
lr_test_mse = mean_squared_error(y_test, lr_predict_y_test) # Mean Squared Error for testing set
lr_test_r2 = r2_score(y_test, lr_predict_y_test)    # R-squared for testing set

# Print the evaluation metrics
print('\t## LINEAR REGRESSION MODEL PERFORMANCE ##')
print('LR MSE (Train): ', lr_train_mse , ">> good value is close to 0")
print('LR R2 (Train): ', lr_train_r2, ">> good value is close to 1")
print('LR MSE (Test): ', lr_test_mse, ">> good value is close to 0")
print('LR R2 (Test): ', lr_test_r2, ">> good value is close to 1")

# Store results in a DataFrame for better visualization
Ir_results = pd.DataFrame(['LR Model', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
# Set column names for the results DataFrame
Ir_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
print(Ir_results)

# Similarly for LR Model, now Random Forest Model
rf = RandomForestRegressor(max_depth=2, random_state=42) # Create an instance of the Random Forest model
rf.fit(X_train, y_train) # Train the model using the training data both features and target variable
# Now the model is trained and can be used to make predictions
rf_predict_y_train = rf.predict(X_train) # Make predictions using training data
rf_predict_y_test = rf.predict(X_test) # Make predictions using testing data

# Calculate evaluation metrics for training and testing sets
rf_train_mse = mean_squared_error(y_train, rf_predict_y_train) # Mean Squared Error for training set
rf_train_r2 = r2_score(y_train, rf_predict_y_train) # R-squared for training set
rf_test_mse = mean_squared_error(y_test, rf_predict_y_test) # Mean Squared Error for testing set
rf_test_r2 = r2_score(y_test, rf_predict_y_test)    # R-squared for testing set

# Print the evaluation metrics
print('\t## RANDOM FOREST MODEL PERFORMANCE ##')
print('RF MSE (Train): ', rf_train_mse , ">> good value is close to 0")
print('RF R2 (Train): ', rf_train_r2, ">> good value is close to 1")
print('RF MSE (Test): ', rf_test_mse, ">> good value is close to 0")
print('RF R2 (Test): ', rf_test_r2, ">> good value is close to 1")

# Store results in a DataFrame for better visualization
rf_results = pd.DataFrame(['RF Model', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
# Set column names for the results DataFrame
rf_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
print(rf_results)

# Combine results of both models for comparison
combined_results = pd.concat([Ir_results, rf_results], ignore_index=True)
print('\t## COMBINED MODEL PERFORMANCE ##')
print(combined_results)

# Data visualization of prediction results for Linear Regression Model
plt.figure(figsize=(5,5)) # Create a new figure for plotting
plt.scatter(x=y_train, y=lr_predict_y_train, c="#00bba7" ,alpha=0.3) # Scatter plot of actual vs predicted values
z = np.polyfit(y_train, lr_predict_y_train, 1) # Fit a polynomial of degree 1 (linear fit)
p = np.poly1d(z) # Create a polynomial function from the fit
plt.plot(y_train, p(y_train), '#7008e7') # Plot the fitted line
plt.ylabel('Predict selling_price') 
plt.xlabel('Experimental selling_price')

# Display the plot
plt.show()