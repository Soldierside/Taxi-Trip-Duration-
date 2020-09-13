#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:51:45 2019

@author: soldierside
"""


import os
import pandas as pd
import numpy as np

url='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/'

os.chdir(url)



data_weather=pd.read_csv('weather_data_nyc_centralpark_2016.csv', delimiter=',')
pd.set_option('display.max_columns', 30)

data_weather.dtypes

### Weather NYC
data_weather.head(4)
data_weather.shape

data_weather['datetime'] = pd.to_datetime(data_weather['date'],format='%d-%m-%Y')
data_weather['pick_day_year']= data_weather['datetime'].dt.dayofyear


data_weather['precipitation'] = np.where(data_weather['precipitation']=='T', '0.00',data_weather['precipitation'])
data_weather['precipitation'] = list(map(float, data_weather['precipitation']))
#si no pongo el list me sale esto "<map object at 0x1a1e0efcc0>". 
#Float es porque necesito los decimales ya que estamos hablando de precipitaciones. 
#Map para aprender a usarlo a raiz de los problemas que tuve con el for.

data_weather['snow fall'] = np.where(data_weather['snow fall']=='T', '0.00',data_weather['snow fall'])
data_weather['snow fall'] = list(map(float, data_weather['snow fall']))

data_weather['snow depth'] = np.where(data_weather['snow depth']=='T', '0.00',data_weather['snow depth'])
data_weather['snow depth'] = list(map(float, data_weather['snow depth']))




