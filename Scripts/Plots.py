#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:52:45 2019

@author: soldierside
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import math



runfile('/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts/Data_Final.py', wdir='/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts')


'''plt.savefig('xgboost.png',bbox_inches='tight')
plt.show()'''


## numeros de pasajeros
p_c=sn.countplot(data_final.vendor_id)
plt.savefig('n_vendedores.png',bbox_inches='tight')

plt.show()


###trip duration

log= np.log(data_final['trip_duration'])

sn.distplot(data_final['trip_duration'])

trip_dur_fig=sn.distplot(log)

plt.savefig('trip_duration.png',bbox_inches='tight')
plt.show()


################

plt.figure(figsize=(11,8))
sn.set(style="white", palette="muted", color_codes=False)
sn.set_context("notebook",font_scale=1.5)
sn.barplot(data=data_final,x='pick_day_week',y='trip_duration',hue='parts_day')
plt.savefig('grafico_trip_pick_day.png',bbox_inches='tight')

plt.show()
    

data_final.columns
##################
## pick_hour
sn.pointplot(data=data_final, x='parts_days', y='trip_duration', palette='muted')#  en la x pick_hour
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Time day')
plt.savefig('pick_hour.png',bbox_inches='tight')

plt.show()

############

sn.barplot(data=data_def, x='pick_day_week',y='trip_duration')



plt.figure(figsize=(11,8))
sn.set_style("ticks")
sn.lineplot(x="pick_hour", y="trip_duration", hue="pick_day_week",data=data_def)

############################

resumen_horas=pd.DataFrame(data_final.groupby(['pick_day_week','pick_hour'])['trip_duration'].mean())
resumen_horas.reset_index(inplace = True)
resumen_horas['unit']=1


plt.figure(figsize=(11,8))
sn.set(style="white", palette="muted", color_codes=False)
sn.set_context("notebook",font_scale=1.5)
sn.tsplot(data=resumen_horas, time="pick_hour", unit = "unit", condition="pick_day_week", value="trip_duration")
plt.savefig('resumen_partes_dia.png',bbox_inches='tight')
plt.show()

######################

resumen_jornada=pd.DataFrame(data_final.groupby(['work_rest_day','pick_hour'])['trip_duration'].mean())
resumen_jornada.reset_index(inplace = True)
resumen_jornada['unit']=1

plt.figure(figsize=(11,8))
sn.set(style="white", palette="muted", color_codes=False)
sn.set_context("notebook",font_scale=1.5)
sn.tsplot(data=resumen_jornada, time="pick_hour", unit = "unit", condition="work_rest_day", value="trip_duration")
plt.savefig('resumen_jornada.png',bbox_inches='tight')
plt.show()

###Nueva york
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.scatter(data_final['pickup_longitude'], data_final['pickup_latitude'],color='blue',s=1, alpha=0.1)

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.savefig('pick_newyork_1.png',bbox_inches='tight')

plt.show()


plt.scatter(data_final['dropoff_longitude'], data_final['dropoff_latitude'],color='blue', s=1, alpha=0.1)
plt.savefig('drop_newyork2.png',bbox_inches='tight')

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


####

sn.countplot(data_final.pick_hour)
plt.savefig('horas_.png',bbox_inches='tight')
plt.show()


sn.catplot(x="vendor_id", y="trip_duration",kind="strip",data=data_final)
plt.savefig('vendor_id.png',bbox_inches='tight')
plt.show()


sn.pointplot(data=data_final, x='pick_day_week',y='trip_duration')
plt.savefig('pick_day_week.png',bbox_inches='tight')
plt.show()

data_final.columns
##Kmeans


##nueva york con 5 clusters

plt.scatter(data_def['pickup_longitude'], data_def['pickup_latitude'],c=data_def['kmeans_pick_5c'], s=1, alpha=0.1,label='colors')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


plt.scatter(data_def['dropoff_longitude'], data_def['dropoff_latitude'],c=data_def['kmeans_drop_5c'], s=1, alpha=0.1)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


##nueva york con 10 clusters
plt.scatter(data_def['pickup_longitude'], data_def['pickup_latitude'],c=data_def['kmeans_pick_10c'], s=1, alpha=0.1)

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()





#Nueva york en plan cutre
plt.scatter(data_def['pca0_pd'], data_def['pca1_pd'],c=data_def['kmeans_pd_5'], s=1, alpha=0.1)

pca_borders = pca_pd_2.transform([[x, y] for x in city_lat_border for y in city_long_border])
plt.xlim(pca_borders[:, 0].min(), pca_borders[:, 0].max())
plt.ylim(pca_borders[:, 1].min(), pca_borders[:, 1].max())
plt.savefig('newyork_cutre.png',bbox_inches='tight')

plt.show()

####Nueva york con PCA

plt.scatter(data_def['pickup_longitude'], data_def['pickup_latitude'],c=data_def['kmeans_pd_5'], s=1, alpha=0.1)

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



plt.scatter(data_def['pickup_longitude'], data_def['pickup_latitude'],c=data_def['kmeans_pd_8c'], s=1, alpha=0.1)

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



plt.scatter(data_def['pickup_longitude'], data_def['pickup_latitude'],c=data_def['kmeans_pd_15c'], s=1, alpha=0.1)

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()





