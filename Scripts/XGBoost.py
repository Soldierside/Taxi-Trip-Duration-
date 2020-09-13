#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:22:55 2019

@author: soldierside
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##data_final
runfile('/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts/Data_Final.py', wdir='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts')


import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_tree


data_final.columns
data_final.drop(['kmeans_pick_10c', 'kmeans_drop_10c', 'kmeans_pick_20c','kmeans_drop_20c','date','maximum temerature', 'minimum temperature', 'average temperature','precipitation', 'snow fall','snow depth','journey_trip_meter','pickup_datetime', 'dropoff_datetime','pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','pickup_datetime_64','pick_day_week','datetime','id','dif_longitude','dif_latitud'],axis=1,inplace=True)



# =============================================================================
# XGBoost
# =============================================================================

data_xgb=data_final.copy()


data_xgb.columns

data_xgb['pick_day_week_name']=data_xgb['pick_day_week_name'].astype('category')

data_xgb['pick_hour']=data_xgb['pick_hour'].astype('category')

data_xgb['parts_day']=data_xgb['parts_day'].astype('category')

data_xgb['work_rest_day']=data_xgb['work_rest_day'].astype('category')

data_xgb['kmeans_pick_5c']=data_xgb['kmeans_pick_5c'].astype('category')

data_xgb['kmeans_drop_5c']=data_xgb['kmeans_drop_5c'].astype('category')

data_xgb['kmeans_pd_5']=data_xgb['kmeans_pd_5'].astype('category')

data_xgb['kmeans_pd_8c']=data_xgb['kmeans_pd_8c'].astype('category')

data_xgb['kmeans_pd_15c']=data_xgb['kmeans_pd_15c'].astype('category')

data_xgb=pd.get_dummies(data_xgb)

data_xgb_train=data_xgb['trip_duration']
data_xgb.drop(['trip_duration'],axis=1,inplace=True)


x_train,x_test,y_train,y_test = train_test_split(data_xgb,data_xgb_train,random_state=10,train_size=0.8)

dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)
dvalid = xgb.DMatrix(x_test, label=y_test)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

parameters = {'min_child_weight': 10, 
            'eta': 0.3, 
            'colsample_bytree': 0.3, 
            'max_depth': 20,
            'subsample': 0.8, 
            'lambda': 1., 
            'nthread': 1, 
            'booster' : 'gbtree', 
            'silent': 1,
            'eval_metric': 'rmse',
            'objective': 'reg:linear',
            'n_jobs':1,
            'learning_rate':0.05}

num_round=300


model=xgb.train(parameters,dtrain,num_round,watchlist,early_stopping_rounds=2,maximize=False, verbose_eval=1)



print('Modeling RMSLE %.5f' % model.best_score)#Modeling RMSLE 460.84473



xgb.plot_importance(model, max_num_features=20, height=0.5)
plt.savefig('xgboost.png',bbox_inches='tight')
plt.show()


data_xgb.to_csv("XBGoost.csv", index= False)
dt=pd.read_csv("XBGoost.csv")
dt.head(2)






