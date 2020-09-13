#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:02:41 2019

@author: soldierside
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



##data_final
runfile('/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts/Data_Final.py', wdir='/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts')


import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


data_final.columns
data_final.drop(['kmeans_pd_8c','kmeans_pd_15c','kmeans_pick_10c', 'kmeans_drop_10c', 'kmeans_pick_20c','kmeans_drop_20c', 'kmeans_pick_50c', 'kmeans_drop_50c','date','maximum temerature', 'minimum temperature', 'average temperature','precipitation', 'snow fall','snow depth','journey_trip_meter','pickup_datetime', 'dropoff_datetime','pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','pickup_datetime_64','pick_day_week','datetime','id','dif_longitude', 'dif_latitud'],axis=1,inplace=True)
#=============================================================================
# #random forest because quiero probar algo nuevo
## M'ha tardat 54 min, no tornar a fer mai més a no ser que tingui temps.
# =============================================================================

data_tree=data_final.copy()


data_tree.dtypes

data_tree['pick_day_week_name']=data_tree['pick_day_week_name'].astype('category')

data_tree['pick_hour']=data_tree['pick_hour'].astype('category')

data_tree['parts_day']=data_tree['parts_day'].astype('category')

data_tree['work_rest_day']=data_tree['work_rest_day'].astype('category')

data_tree['kmeans_pick_5c']=data_tree['kmeans_pick_5c'].astype('category')

data_tree['kmeans_drop_5c']=data_tree['kmeans_drop_5c'].astype('category')

data_tree['kmeans_pd_5']=data_tree['kmeans_pd_5'].astype('category')


data_tree=pd.get_dummies(data_tree)
data_tree.columns

data_tree_train=data_tree['trip_duration']
data_tree.drop(['trip_duration'],axis=1,inplace=True)


x_train,x_test,y_train,y_test = train_test_split(data_tree,data_tree_train,random_state=10,train_size=0.8)

x=x_train
y=y_train


model = RandomForestRegressor()

model.get_params()
params={'bootstrap': [True],
        'criterion':['mse'],
        'max_depth': [40,50],# Maxima pofundidad del arbol
        'max_features': [30, 50], # numero de features a considerar en cada split
        'max_leaf_nodes': [10,20,30], # maximo de nodos del arbol
        'min_impurity_decrease' : [0.05], # un nuevo nodo se harà si al hacerse se decrece la impurity en un threshold por encima del valor
        'min_samples_split': [5,10], # The minimum number of samples required to split an internal node:
        'n_estimators': [50,150] # number of trees
        }

# scoring: lista de metricas a obtener
scoring = ['neg_mean_squared_error','neg_mean_squared_log_error', 'r2','explained_variance']#rmse afegir


grid_solver = GridSearchCV(estimator = model, # model to train
                   param_grid = params, # param_grid
                   scoring = scoring,
                   cv = 3,
                   refit = 'neg_mean_squared_error',
                   verbose = 2,
                   n_jobs=1)

model_result = grid_solver.fit(x,y)

#sorted(sklearn.metrics.SCORERS.keys()) para saber los scores que puedes utilizar, probare de hacer uno con clusters ahora haber como queda

# Mejores parametros
model_result.best_params_
x.shape
'''
{'bootstrap': True,
 'criterion': 'mse',
 'max_depth': 40,
 'max_features': 30,
 'max_leaf_nodes': 30,
 'min_impurity_decrease': 0.05,
 'min_samples_split': 10,
 'n_estimators': 300}
'''
# negative mean square error
# train
model_result.score(x,y)
y_pred=model_result.predict(x)
sklearn.metrics.mean_squared_error(y,y_pred)

rms=np.sqrt(sklearn.metrics.mean_squared_error(y,y_pred))#442.4157


# entrenamos con los mejores parametros
model_result.best_params_
res=model_result.best_estimator_.fit(x,y)
'''
                    bootstrap=True, criterion='mse', max_depth=40,
                      max_features=30, max_leaf_nodes=30,
                      min_impurity_decrease=0.05, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=10,
                      min_weight_fraction_leaf=0.0, n_estimators=300,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False
'''
# train
res.score(x,y)#0.6132 
y_pred=res.predict(x)
sklearn.metrics.mean_squared_error(y,y_pred)#
rms=np.sqrt(sklearn.metrics.mean_squared_error(y,y_pred))#444.46




data_tree.to_csv("Randomforest.csv", index= False)
dt=pd.read_csv("Randomforest.csv")
dt.head(2)
















