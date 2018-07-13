# This program uses the House Prices competition dataset from kaggle.com
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('../input/train.csv')

# Set target column
train_y = train.SalePrice

# Select columns used to train the model
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Training vector
train_X = train[predictor_cols]

# Create a random forest model
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('../input/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]

# Use the model to make predictions
predicted_prices = my_model.predict(test_X)

# Create a solution submission.csv file
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)