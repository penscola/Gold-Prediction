#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# importing the dataset
df = pd.read_csv('/media/penscola/Penscola@Tech/Projects/Gold-Prediction/data/gld_price_data.csv')
print("Data Imported Successfully ✅")

# Removing the date column
df = df.drop(columns =['Date'],axis=1)

# splitting the dataset
X = df.drop(columns=['GLD'],axis=1)
y = df['GLD']
X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.2 , random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train,y_train)
X_test_scaled  =scaler.transform(X_test)

# model
rf1 = RandomForestRegressor()

# Hyperparameter Tuning
n_estimators = [20,60,100,120]
max_features = [0.2,0.6,1.0]
max_depth = [2,8,None]
max_samples = [0.5,0.75,1.0]

param_grid = { 'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'max_samples' : max_samples
}
rf1_grid = GridSearchCV(estimator = rf1,
                       param_grid = param_grid,
                       cv=5,
                       verbose =2,
                       n_jobs = -1)

# training the model
rf1_grid.fit(X_train_scaled,y_train)

# saving the model
output_file = '/media/penscola/Penscola@Tech/Projects/Gold-Prediction/model/Random-Forest-Regressor.pkl'

with open(output_file, 'wb') as f_out:
    pickle.dump((scaler, rf1_grid), f_out)