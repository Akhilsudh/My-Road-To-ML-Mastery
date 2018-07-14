# This program uses the House Prices competition dataset from kaggle.com
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Read the data
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

# Select Predictors
trainData.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
trainy = trainData.SalePrice
trainX = trainData.drop(['SalePrice'], axis = 1).select_dtypes(exclude = ['object'])

testX = testData.select_dtypes(exclude = ['object'])

# Impute NaN columns
myImputer = SimpleImputer()
train_X = myImputer.fit_transform(trainX)
test_X = myImputer.transform(testX)

# Fit model
my_model = XGBRegressor()
my_model.fit(train_X, trainy, verbose = False)

# Make prediction
prediction = my_model.predict(test_X)

# Make result submission file
my_submission = pd.DataFrame({'Id': testData.Id, 'SalePrice': prediction})
my_submission.to_csv('submission.csv', index=False)