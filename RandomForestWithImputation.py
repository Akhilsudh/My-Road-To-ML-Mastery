# This program uses the House Prices competition dataset from kaggle.com
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Read the data
train = pd.read_csv('../input/train.csv')
train_y = train.SalePrice
predictor_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]

# Impute NaN values for better results
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(train_X)

# Create a random forest model
my_model = RandomForestRegressor()
my_model.fit(imputed_X_train, train_y)

# Read the test data
test = pd.read_csv('../input/test.csv')
test_X = test[predictor_cols]
imputed_X_test = my_imputer.transform(test_X)

# Use the model to make predictions
predicted_prices = my_model.predict(imputed_X_test)

# Create a solution submission.csv file
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)