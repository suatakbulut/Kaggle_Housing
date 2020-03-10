"""
Created on Sat Jan 25 18:05:52 2020

@author: suat.akbulut

Description: This code tries to predict house prices featured on test.csv 
data set by getting trained on some part of the train.csv data, tested on 
the remaining part of it. And, finally evaluated by the competition holders 
on test.csv data. It compares 5 different prediction techniques without 
optimizing the default hyper parameters of these models and compare them


"""
import pandas as pd
import numpy as np # I am not sure if I will need numpy, however, out of habit I import in anyway

# I am importing WARNINGS class to suppress warnings
import warnings
warnings.filterwarnings('ignore')

# load train and test data as train_data and test_data
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

# If SalePrice is missing for some observation, drop those observations
train_data.dropna(axis=0, subset=["SalePrice"],  inplace=True) 

y = train_data.SalePrice
train_data.drop(["SalePrice"],axis=1, inplace=True)

# Determine the numeric and categorical column names
numerical_cols   = [col for col in train_data.columns if train_data[col].dtype in ["int64", "float"]]

# If a variable has more than max_categories, I will drop that variable.
max_categories = 15
categorical_cols = [col for col in train_data.columns if train_data[col].dtype == "object" and train_data[col].nunique() < max_categories]
my_cols          = numerical_cols + categorical_cols

# We will only work with numeric and categorical columns. 
X = train_data[my_cols]
X_test = test_data[my_cols]

# Now we have our dependend and independent variables, y and X
# Let us split the them into train and validation parts
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Handle the missing values: 

from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing Numerical data
numerical_transformer = SimpleImputer(strategy = "median")

# Preprocessing Categorical data
categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

preprocessor = ColumnTransformer(
        transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
                ])

"""
At this stage we are done with preprocessing. 
Combine preprocessing with our selection of models using a pipeline
with the default hyper-parameters and compare them based on
their mean absolute error. 
"""
from sklearn.metrics import mean_absolute_error 


# ------------------------
# 1) Random Forest Model: 
# ------------------------
from sklearn.ensemble import RandomForestRegressor 
forest_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor())
        ])
# fit the train data to the model
forest_pipeline.fit(X_train, y_train)

# Measure the score of our model    
forest_y_hat = forest_pipeline.predict(X_valid)
forest_score = mean_absolute_error(y_valid, forest_y_hat)

# predict the values in test data
forest_preds = forest_pipeline.predict(X_test)


# -----------------
# 2) XGB Model: 
# -----------------
from xgboost import XGBRegressor
xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor() )
        ])
# fit the train data to the model
xgb_pipeline.fit(X_train, y_train)

# Measure the score of our model    
xgb_y_hat = xgb_pipeline.predict(X_valid)
xgb_score = mean_absolute_error(y_valid, xgb_y_hat)

# predict the values in test data
xgb_preds = xgb_pipeline.predict(X_test)


# -----------------
# 3) SVM Model: 
# -----------------
from sklearn import svm
svm_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", svm.SVC() )
        ])
# fit the train data to the model
svm_pipeline.fit(X_train, y_train)

# Measure the score of our model    
svm_y_hat = svm_pipeline.predict(X_valid)
svm_score = mean_absolute_error(y_valid, svm_y_hat)

# predict the values in test data
svm_preds = svm_pipeline.predict(X_test)


# -----------------
# 4) kNN Model: 
# -----------------
from sklearn.neighbors import KNeighborsClassifier
kNN_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier() )
        ])
# fit the train data to the model
kNN_pipeline.fit(X_train, y_train)

# Measure the score of our model    
kNN_y_hat = kNN_pipeline.predict(X_valid)
kNN_score = mean_absolute_error(y_valid, kNN_y_hat)

# predict the values in test data
kNN_preds = kNN_pipeline.predict(X_test)


# -----------------
# 5) OLS Model: 
# -----------------
from sklearn.linear_model import LinearRegression
ols_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier() )
        ])
# fit the train data to the model
ols_pipeline.fit(X_train, y_train)

# Measure the score of our model    
ols_y_hat = ols_pipeline.predict(X_valid)
ols_score = mean_absolute_error(y_valid, ols_y_hat)

# predict the values in test data
ols_preds = ols_pipeline.predict(X_test)

# -----------------   Results   -----------------

models = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'XGB', 'Random Forest', 'OLS' ],
    'Score': [svm_score, kNN_score, xgb_score, forest_score, ols_score ]})
results = models.sort_values(by='Score').reset_index(drop=True)

print(results)
print("\tThe winner is {} Model!..".format(results.loc[0].Model))

#-------------------------------------------------------------------------------
# Create the predictions of the test file as a csv, called submission.csv, 
# which can be uploaded as solution to the competition
#-------------------------------------------------------------------------------
output = pd.DataFrame({'Id': X_test.Id,
                       'SalePrice': xgb_preds})
output.to_csv("submission.csv")
print("submission.csv file has been created.")