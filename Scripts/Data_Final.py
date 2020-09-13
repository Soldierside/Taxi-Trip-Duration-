#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:17:10 2019

@author: soldierside
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import math
from math import sin, cos, sqrt, atan2, radians

os.getcwd()

directorio='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts'

#enlace data_weather
runfile('/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts/Data_Weather.py', wdir='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts')
data_weather.head(2)

#enlace del data_train
runfile('/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts/Data_train.py', wdir='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts')

data_train.head(2)

data_def = pd.merge(data_train, data_weather, on= 'pick_day_year', how = 'left')



import sklearn

from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score




#neighborhood_NY

# para clusters, son los barrios de nueva york, kmeans
### standar scaler



#CLUSTERS


#per cada pick up i dropp  fer un distancia euclidial per fer les variables de barri d'arribada i sortida. 

#PCA distancies


pca_pd = data_def[['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude']]


#pca_pd= StandardScaler().fit_transform(data_def[['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']])


pca_pd_2= PCA(n_components=3).fit(pca_pd)

data_def.columns
pca_pd_2.explained_variance_ratio_
data_def['pca0_pd'] = pca_pd_2.transform(pca_pd)[:, 0]
data_def['pca1_pd'] = pca_pd_2.transform(pca_pd)[:, 1]
data_def['pca2_pd'] = pca_pd_2.transform(pca_pd)[:, 2]


data_def.columns


#PCA temperatura

pca_w = data_def[['maximum temerature', 'minimum temperature', 'average temperature','precipitation', 'snow fall', 'snow depth']]

pca_w_2= PCA(n_components=2).fit(pca_w)
pca_w_2.explained_variance_ratio_


data_def['pca_w0'] = pca_w_2.transform(pca_w)[:, 0]




data_def.dtypes



#kmeans


data_def.columns

data_km=data_def.loc[:,['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude']]


kmeans = KMeans(n_clusters=5, random_state=10)
kmeans2=kmeans.fit(data_km)
kmeans2.cluster_centers_ = neighborhood_NY



data_def['kmeans_pick_5c'] = kmeans2.predict(data_def[['pickup_latitude', 'pickup_longitude']])





data_def['kmeans_drop_5c'] = kmeans2.predict(data_def[['dropoff_latitude', 'dropoff_longitude']])






#formula con 10 clusters

pick_drop_10 = np.vstack((data_def[['pickup_latitude', 'pickup_longitude']],data_def[['dropoff_latitude', 'dropoff_longitude']]))

kmeans_10c = MiniBatchKMeans(n_clusters=10, random_state=10).fit(pick_drop_10)

data_def['kmeans_pick_10c'] = kmeans_10c.predict(data_def[['pickup_latitude', 'pickup_longitude']])

data_def['kmeans_drop_10c'] = kmeans_10c.predict(data_def[['dropoff_latitude', 'dropoff_longitude']])






##forumula con 20 clusters

pick_drop_20 = np.vstack((data_def[['pickup_latitude', 'pickup_longitude']],data_def[['dropoff_latitude', 'dropoff_longitude']]))

kmeans_20c = MiniBatchKMeans(n_clusters=20, random_state=10).fit(pick_drop_20)

data_def['kmeans_pick_20c'] = kmeans_20c.predict(data_def[['pickup_latitude', 'pickup_longitude']])

data_def['kmeans_drop_20c'] = kmeans_20c.predict(data_def[['dropoff_latitude', 'dropoff_longitude']])






#cosas randoms que he encontrado, me da los kilometros de distancia entre los clusters

data_def['pick_km_dist'] =  data_def.apply( lambda x: lat_lon_converter(  x['pickup_latitude'], x['pickup_longitude'], neighborhood_NY[x['kmeans_pick_5c']][0],neighborhood_NY[x['kmeans_pick_5c']][1],'km'),axis=1)
 
data_def['drop_km_dist'] =  data_def.apply( lambda x: lat_lon_converter(  x['dropoff_latitude'], x['dropoff_longitude'], neighborhood_NY[x['kmeans_drop_5c']][0], neighborhood_NY[x['kmeans_drop_5c']][1],'km'), axis=1)


data_def['pick_drop_km_dist'] =  data_def.apply(lambda x:lat_lon_converter(neighborhood_NY[x['kmeans_pick_5c']][0], neighborhood_NY[x['kmeans_pick_5c']][1],                                           neighborhood_NY[x['kmeans_drop_5c']][0], neighborhood_NY[x['kmeans_drop_5c']][1], 'km'), axis=1 )



#######

data_pca=data_def.loc[:,['pca0_pd', 'pca1_pd', 'pca2_pd']]


kmeans_5_pca = KMeans(n_clusters=5, random_state=10)
kmeans_pca_5=kmeans_5_pca.fit(data_pca)


data_def['kmeans_pd_5'] = kmeans_pca_5.predict(data_pca)




kmeans_8 = KMeans(n_clusters=8, random_state=10)
kmeans_pca_8=kmeans_8.fit(data_pca)


data_def['kmeans_pd_8c'] = kmeans_pca_8.predict(data_pca)






kmeans_15 = KMeans(n_clusters=15, random_state=10)
kmeans_pca_15=kmeans_15.fit(data_pca)


data_def['kmeans_pd_15c'] = kmeans_pca_15.predict(data_pca)


data_def.columns



data_final=data_def.copy()




