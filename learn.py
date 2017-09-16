# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 13:34:32 2017

@author: zx_pe
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pickle.load(open('dataarray.p','rb'))
#%%
def stripNA (vec, replace = 0, fun = np.nanmean, cts = False):
    if cts:
        r = fun(vec)
    else:
        r = replace 
    g = vec.apply(lambda x: r if np.isnan(x) else x)
    return g

def validator(estimator, param_grid, X, Y):
    clf = GridSearchCV(estimator, param_grid)
    clf.fit(X,Y)
    print(clf.best_params_)
    print(clf.best_score_)
    return clf
#%%
class EnsembleTrainer:
    def __init__(self, l1models, l2model, param_grids = None):
        
        self.l1models = l1models
        self.l2model = l2model
        
    def tunel1(self, data, labels, param_grids):
        self.y_train = labels
        
        self.tunedModels = []
        for i in range(len(self.l1models)):
            self.tunedModels.append(validator(self.l1models[i](), param_grids[i], data, labels))    
            print('Trained model' + str(i))
        
        self.l1score = [model.best_score_ for model in self.tunedModels]
        
        ed = [model.predict(data) for model in self.tunedModels]
        self.l1TOutput = np.array(ed).transpose()
        
    def definel1(self, data, labels, param_grids):
        self.y_train = labels
        
        self.tunedModels = []
        for i in range(len(self.l1models)):
            self.tunedModels.append(self.l1models[i](**param_grids[i]))
            
        for model in self.tunedModels:
            model.fit(data, labels)
            print('Trained')
        
        print('Making l1 Predictions')
        ed = [model.predict(data) for model in self.tunedModels]
        self.l1TOutput = np.array(ed).transpose()
        
    def predictl1(self, data):
        ed = [model.predict(data) for model in self.tunedModels]
        self.l1Output = np.array(ed).transpose()
        
        
    def tunel2(self, param_grid):
        self.tunedl2 = validator(self.l2model(), param_grid, self.l1TOutput, self.y_train)
        self.l2score = self.tunedl2.best_score_
        self.tunedl2.fit(self.l1TOutput, self.y_train)
        
    def definel2(self, param_grid):
        self.tunedl2 = self.l2model(**param_grid)
        self.tunedl2.fit(self.l1TOutput, self.y_train)
        
    def predictl2(self):
        self.l2Output = self.tunedl2.predict(self.l1Output)
        
    def fit(self, X_train, Y_train, param_gridsl1, param_gridsl2, tune = True):
        if tune:
            self.tunel1(X_train, Y_train, param_gridsl1)
            self.tunel2(param_gridsl2)
        else:
            self.definel1(X_train,Y_train, param_gridsl1)
            self.definel2(param_gridsl2)

    def predict(self, X_test):
        self.predictl1(X_test)
        self.predictl2()
        
    def evaluatel2(self, Y_test):
        return metrics.mean_squared_error(self.l2Output, Y_test)
        
    def evaluatel1(self, Y_test):
        return np.apply_along_axis(metrics.mean_squared_error, 0, self.l1Output, Y_test)
#%%
X = df.drop(['rainfall','datestamp','SID'], axis = 1)
X = preprocessing.scale(X.apply(stripNA, axis = 0, cts=True).values)

Y = df.sum(axis=1).values
Y = df['rainfall'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=2)
#%%
tree_grid = {
        'max_features': [5, X.shape[1]],
        'n_estimators': [10, 100, 10],
        'max_leaf_nodes': [2,100,5]
}

knn_grid = {
        'n_neighbors': [5, X.shape[0], 1000],
}

lgb_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

lasso_grid = {
        'alpha': [0,1,0.1],
}
param_grids = [tree_grid, tree_grid, knn_grid, lgb_grid]
#%%
ET = EnsembleTrainer([RandomForestRegressor, ExtraTreesRegressor,KNeighborsRegressor, lgb.LGBMRegressor], Lasso)
ET.fit(X_train, y_train, param_grids, lasso_grid, tune=True)
ET.predict(X_test)
ET.evaluatel1(y_test)
ET.evaluatel2(y_test)
#%%
tree_grid = {
        'max_features': 5,
        'n_estimators': 100,
        'max_leaf_nodes': 100
}

knn_grid = {
        'n_neighbors': 3000,
}

lgb_grid = {
    'learning_rate': 0.4,
    'n_estimators': 35
}

lasso_grid = {
        'alpha': 0.4
}
param_grids = [tree_grid, tree_grid, knn_grid, lgb_grid] 
#%%
ET = EnsembleTrainer([RandomForestRegressor, ExtraTreesRegressor, KNeighborsRegressor, lgb.LGBMRegressor], Lasso)
ET.fit(X_train, y_train, param_grids, lasso_grid, tune=False)
ET.predict(X_test)
ET.evaluatel1(y_test)
ET.evaluatel2(y_test)

#%%
cols = df.drop(['rainfall','datestamp','SID'], axis = 1).columns
features = [sorted(zip(cols, i.feature_importances_), key = lambda x: x[1], reverse=False) for i in ET.tunedModels[0:2]]
impt = [[feature [1] for feature in l] for l in features]
col = [[feature [0] for feature in l] for l in features]

plt.subplots(figsize=(12, 8))
sns.barplot(x=impt[0], y=col[0], orient = 'h')

plt.subplots(figsize=(12, 8))
sns.barplot(x=impt[1], y=col[1], orient = 'h')

features = sorted(zip(cols, ET.tunedModels[3].feature_importances_), key = lambda x: x[1], reverse=False)
impt = [feature [1] for feature in features]
col = [feature [0] for feature in features]

plt.subplots(figsize=(12, 8))
sns.barplot(x=impt, y=col, orient = 'h')