#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:47:16 2019

@author: soldierside
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import math

#funciones
runfile('/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts/Funciones.py', wdir='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts')






url='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/'


os.chdir(url)
os.listdir(url)




# =============================================================================
# 
# =============================================================================

data_taxi=pd.read_csv('train.csv',delimiter=',')
pd.set_option('display.max_columns', 30)

data_taxi.dtypes

# =============================================================================
# #necesito
# para este dataset predecir el tiempo de recorrido. 
# Ideas: tiempo de transito, dia del año, ubicación de la distancia, hora del dia. =============================================================================

# =============================================================================
# #tratamiento de datos con otro data
# =============================================================================

data_train=data_taxi.copy()

data_train=data_train.iloc[0:100000,:]

data_train.dtypes



data_train=data_train.drop(data_train[data_train['trip_duration'] < 60].index)#de esta forme me puedo cargar 8 mil lineas inutiles

data_train=data_train.drop(data_train[data_train['trip_duration'] > 50400].index)#14 horas de taxi como mucho, lo pensare mejor. 

data_train['dif_longitude']=data_train['pickup_longitude']-data_train['dropoff_longitude']

data_train['dif_latitud']=data_train['pickup_latitude']-data_train['dropoff_latitude']




data_train['pickup_datetime_64'] = pd.to_datetime(data_train.pickup_datetime)
data_train['pick_day'] = data_train['pickup_datetime_64'].dt.day
data_train['pick_hour']=data_train['pickup_datetime_64'].dt.hour
data_train['pick_week_of_year'] = data_train['pickup_datetime_64'].dt.weekofyear
data_train['pick_day_year'] = data_train['pickup_datetime_64'].dt.dayofyear

data_train['pick_day_week'] = data_train['pickup_datetime_64'].dt.dayofweek#el 0 es lunes y el 6 es domingo por lo tanto esta la tengo que modificar.

data_train['pick_day_week_name']=data_train['pick_day_week'].replace({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thusday',4:'Friday',5:'Saturday',6:'Sunday'})

#necesito sacar el nuevo dropoof y compararlo 
#llegados a este punto puedo cargarme el dropoff datetime

#pasar el valor de trip_duration a minutos y entonces sumarlo, creo que será más rapido


data_train['parts_day']=data_train['pick_hour'].replace({
           0:'Evening',
           1:'Late night',
           2:'Late night',
           3:'Late night',
           4:'Late night',
           5:'Late night',
           6:'Morning',
           7:'Morning',
           8:'Morning',
           9:'Morning',
           10:'Morning',
           11:'Morning',
           12:'Morning',
           13:'Midday',
           14:'Midday',
           15:'Midday',
           16:'Afternoon',
           17:'Afternoon',
           18:'Afternoon',
           19:'Afternoon',
           20:'Afternoon',
           21:'Evening',
           22:'Evening',
           23:'Evening'})
 


          

data_train['work_rest_day']=data_train['pick_day_week'].replace({
        0:'workday',
        1:'workday',
        2:'workday',
        3:'workday',
        4:'workday',
        5:'rest_day',
        6:'rest_day',})          






##Barrios de Nueva York
Staten_Island=40.56233, -74.13986
Bronx=40.84985, -73.86641
Queens=40.742054,-73.769417
Brooklyn=40.650002,-73.949997
Manhattan=40.78343, -73.96625 #mide 22 km de largo y 3,6 de ancho


neighborhood_NY=np.array(
        [[40.579021,-74.151535],
        [40.837049,-73.865429],
        [40.742054,-73.769417],
        [40.650002,-73.949997],
        [40.78343,-73.96625]]
        )


data_train.columns


# data_train.drop(columns='dis_SI',inplace=True)
data_train['dis_SI_pick']=data_train.apply(lambda x: haversine_pick(x, 40.56233, -74.13986), axis=1)
data_train['dis_Bronx_pick']=data_train.apply(lambda x: haversine_pick(x, 40.84985, -73.86641), axis=1)
data_train['dis_Queens_pick']=data_train.apply(lambda x: haversine_pick(x, 40.742054,-73.769417), axis=1)
data_train['dis_Brooklyn_pick']=data_train.apply(lambda x: haversine_pick(x, 40.650002,-73.949997), axis=1)
data_train['dis_Manh_pick']=data_train.apply(lambda x: haversine_pick(x, 40.78343, -73.96625), axis=1)

data_train['dis_SI_drop']=data_train.apply(lambda x: haversine_drop(x, 40.56233, -74.13986), axis=1)
data_train['dis_Bronx_drop']=data_train.apply(lambda x: haversine_drop(x, 40.84985, -73.86641), axis=1)
data_train['dis_Queens_drop']=data_train.apply(lambda x: haversine_drop(x, 40.742054,-73.769417), axis=1)
data_train['dis_Brooklyn_drop']=data_train.apply(lambda x: haversine_drop(x, 40.650002,-73.949997), axis=1)
data_train['dis_Manh_drop']=data_train.apply(lambda x: haversine_drop(x, 40.78343, -73.96625), axis=1)

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85) 




data_train['journey_trip']=haversine_(data_train['pickup_latitude'],data_train['pickup_longitude'],data_train['dropoff_latitude'],data_train['dropoff_longitude'])#esta en kilometros

data_train['journey_trip_meter']=data_train['journey_trip'].apply(lambda x: x*1000)

data_train['journey_trip_meter']=data_train['journey_trip_meter'].apply(lambda x: int(x))

data_train.dtypes
data_train.groupby('vendor_id')['id'].count()
data_train.groupby('vendor_id').mean()


data_train['manhattan_pick_drop'] = manhattan_distance_pd(data_train['pickup_latitude'], data_train['pickup_longitude'], data_train['dropoff_latitude'], data_train['dropoff_longitude'])



data_train.shape
data_train.dtypes

data_train.head(2)

data_train

