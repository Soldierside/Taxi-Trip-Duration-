#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:53:34 2019

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


#enlace data_weather
runfile('/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts/Data_Weather.py', wdir='/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts')

data_weather.head(2)

#enlace del data_train
runfile('/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts/Data_train.py', wdir='/Users/soldierside/Documents/Bootcamp/Projecte final/nyc-taxi-trip-duration/Scripts')

#posar tambe la rmsle aqui i al random forest

data_def = pd.merge(data_train, data_weather, on= 'pick_day_year', how = 'left')



from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data_def.drop(['id','pickup_datetime', 'dropoff_datetime','dif_longitude', 'dif_latitud','date','datetime','pickup_datetime_64','journey_trip_meter'],axis=1,inplace=True)

data_def.dtypes
data_def.columns
# =============================================================================
# ##Regresion lineal con la distancia solo
# =============================================================================


data_regresion=data_def.copy()
data_regresion=data_def.drop(['vendor_id','store_and_fwd_flag','passenger_count','pick_day', 'pick_hour', 'pick_week_of_year',
       'pick_day_year', 'pick_day_week', 'pick_day_week_name', 'parts_day',
       'work_rest_day', 'dis_SI_pick', 'dis_Bronx_pick', 'dis_Queens_pick',
       'dis_Brooklyn_pick', 'dis_Manh_pick', 'dis_SI_drop', 'dis_Bronx_drop',
       'dis_Queens_drop', 'dis_Brooklyn_drop', 'dis_Manh_drop', 'journey_trip',
       'manhattan_pick_drop', 'distance', 'maximum temerature',
       'minimum temperature', 'average temperature', 'precipitation',
       'snow fall', 'snow depth'],axis=1)



data_train=data_regresion['trip_duration']
data_regresion=data_regresion.drop(['trip_duration'],axis=1)
data_regresion.dtypes


x_train,x_test,y_train,y_test = train_test_split(data_regresion,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train



#Ajusto modelo
reg = linear_model.LinearRegression()
reg = reg.fit(x,y)
print('Intercept: \n', reg.intercept_)
print('Slope: \n', reg.coef_)

#Calculo errores y medidas riesgo
yhat = reg.predict(x)
print("Mean squared error: %.2f" % mean_squared_error(y, yhat))
print('R2: %.2f' % r2_score(y, yhat)) #R2 del 0.19, osea es basura, tengo demasiada dispersion.
rms = sqrt(mean_squared_error(y, yhat))#rms de 654

plt.scatter(y, yhat)
plt.plot(x,yhat, color='red', linewidth=3)
plt.show()

# =============================================================================
# Regresion lineal 2 con distancia y tiempo
# =============================================================================

data_regresion2=data_def.copy()

data_regresion2=data_regresion2.drop(['vendor_id','store_and_fwd_flag','passenger_count','dis_SI_pick', 'dis_Bronx_pick', 'dis_Queens_pick',
       'dis_Brooklyn_pick', 'dis_Manh_pick', 'dis_SI_drop', 'dis_Bronx_drop',
       'dis_Queens_drop', 'dis_Brooklyn_drop', 'dis_Manh_drop', 'journey_trip',
       'manhattan_pick_drop', 'distance', 'maximum temerature',
       'minimum temperature', 'average temperature', 'precipitation',
       'snow fall', 'snow depth','pick_day_week'],axis=1)

#pasar a categorido el pick_day_week y el pick_hour.


data_regresion2['pick_hour']=data_regresion2['pick_hour'].astype('category')

data_regresion2['pick_day_week_name']=data_regresion2['pick_day_week_name'].astype('category')
data_regresion2['parts_day']=data_regresion2['parts_day'].astype('category')
data_regresion2['work_rest_day']=data_regresion2['work_rest_day'].astype('category')


data_regresion2=pd.get_dummies(data_regresion2)


data_train=data_regresion2['trip_duration']
data_regresion2=data_regresion2.drop(['trip_duration'],axis=1)
data_regresion2.dtypes


x_train,x_test,y_train,y_test = train_test_split(data_regresion2,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train






#Ajusto modelo
reg = linear_model.LinearRegression()
reg = reg.fit(x,y)
print('Intercept: \n', reg.intercept_)
print('Slope: \n', reg.coef_)

#Calculo errores y medidas riesgo
yhat = reg.predict(x)
print("Mean squared error: %.2f" % mean_squared_error(y, yhat))
print('R2: %.2f' % r2_score(y, yhat)) #R2 del 0.21, ha mejorado 0.09 puntos

rms = sqrt(mean_squared_error(y, yhat))#rms de 644


plt.scatter(y, yhat)
plt.plot(x,yhat, color='red', linewidth=3)
plt.show()





# =============================================================================
# ##regressio lineal 3 
# no le quitaremos el vendor id, todavia no pondré el tiempo ahora pasare a las distancias uqe he sacado =============================================================================

data_regresion3=data_def.copy()

data_regresion3=data_regresion3.drop(['store_and_fwd_flag','passenger_count','maximum temerature',
       'minimum temperature', 'average temperature', 'precipitation',
       'snow fall', 'snow depth','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis=1)


data_regresion3['pick_hour']=data_regresion3['pick_hour'].astype('category')

data_regresion3['pick_day_week_name']=data_regresion3['pick_day_week_name'].astype('category')
data_regresion3['parts_day']=data_regresion3['parts_day'].astype('category')
data_regresion3['work_rest_day']=data_regresion3['work_rest_day'].astype('category')
data_regresion3['vendor_id']=data_regresion3['vendor_id'].astype('category')


data_regresion3=pd.get_dummies(data_regresion3)


data_train=data_regresion3['trip_duration']
data_regresion3=data_regresion3.drop(['trip_duration'],axis=1)
data_regresion3.dtypes



x_train,x_test,y_train,y_test = train_test_split(data_regresion3,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train



#Ajusto modelo
reg = linear_model.LinearRegression()
reg = reg.fit(x,y)
print('Intercept: \n', reg.intercept_)
print('Slope: \n', reg.coef_)

#Calculo errores y medidas riesgo
yhat = reg.predict(x)
print("Mean squared error: %.2f" % mean_squared_error(y, yhat))
print('R2: %.2f' % r2_score(y, yhat)) #0.52 ha millorat una barbaritat a l'hora de possar totes les distancies
rms = sqrt(mean_squared_error(y, yhat))#rms de 493

plt.scatter(y, yhat)
plt.plot(x,yhat, color='red', linewidth=3)
plt.show()



# =============================================================================
# Regressio lineal 4 
#Possare la temperatura i el nombre de passatgers =============================================================================

data_regresion4=data_def.copy()

data_regresion4.drop(['store_and_fwd_flag'
       ,'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


data_regresion4['pick_hour']=data_regresion4['pick_hour'].astype('category')

data_regresion4['pick_day_week_name']=data_regresion4['pick_day_week_name'].astype('category')
data_regresion4['parts_day']=data_regresion4['parts_day'].astype('category')
data_regresion4['work_rest_day']=data_regresion4['work_rest_day'].astype('category')
data_regresion4['vendor_id']=data_regresion4['vendor_id'].astype('category')
data_regresion4['passenger_count']=data_regresion4['passenger_count'].astype('category')


data_regresion4=pd.get_dummies(data_regresion4)


data_train=data_regresion4['trip_duration']
data_regresion4=data_regresion4.drop(['trip_duration'],axis=1)
data_regresion4.dtypes



x_train,x_test,y_train,y_test = train_test_split(data_regresion4,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train


#Ajusto modelo
reg = linear_model.LinearRegression()
reg = reg.fit(x,y)
print('Intercept: \n', reg.intercept_)
print('Slope: \n', reg.coef_)

#Calculo errores y medidas riesgo
yhat = reg.predict(x)
print("Mean squared error: %.2f" % mean_squared_error(y, yhat))
print('R2: %.2f' % r2_score(y, yhat)) #0.54, ha millorat poc, el temps no te gaire pes com hem pogut veure
rms = sqrt(mean_squared_error(y, yhat))#rms de 492

plt.scatter(y, yhat)
plt.plot(x,yhat, color='red', linewidth=3)
plt.show()



# =============================================================================
# Regresion lineal con las dummys bien tratadas
# =============================================================================


data_regresion5=data_def.copy()

data_regresion5.drop(['store_and_fwd_flag'
       ,'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


data_regresion5['pick_hour']=data_regresion5['pick_hour'].astype('category')

data_regresion5['pick_day_week_name']=data_regresion5['pick_day_week_name'].astype('category')
data_regresion5['parts_day']=data_regresion5['parts_day'].astype('category')
data_regresion5['work_rest_day']=data_regresion5['work_rest_day'].astype('category')
data_regresion5['vendor_id']=data_regresion5['vendor_id'].astype('category')
data_regresion5['passenger_count']=data_regresion5['passenger_count'].astype('category')


data_regresion5=pd.get_dummies(data_regresion4)


data_train=data_regresion5['trip_duration']
data_regresion5=data_regresion5.drop(['trip_duration'],axis=1)
data_regresion5.dtypes



x_train,x_test,y_train,y_test = train_test_split(data_regresion5,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train


#Ajusto modelo
reg = linear_model.LinearRegression()
reg = reg.fit(x,y)
print('Intercept: \n', reg.intercept_)
print('Slope: \n', reg.coef_)

#Calculo errores y medidas riesgo
yhat = reg.predict(x)
print("Mean squared error: %.2f" % mean_squared_error(y, yhat))
print('R2: %.2f' % r2_score(y, yhat)) 
rms = sqrt(mean_squared_error(y, yhat))#

plt.scatter(y, yhat)
plt.plot(x,yhat, color='red', linewidth=3)
plt.show()


# =============================================================================
# Regressio polonomial de nivell 2
# =============================================================================
data_polinomial=data_def.copy()

data_polinomial.drop(['store_and_fwd_flag'
       ,'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


data_polinomial['pick_hour']=data_polinomial['pick_hour'].astype('category')

data_polinomial['pick_day_week_name']=data_polinomial['pick_day_week_name'].astype('category')
data_polinomial['parts_day']=data_polinomial['parts_day'].astype('category')
data_polinomial['work_rest_day']=data_polinomial['work_rest_day'].astype('category')
data_polinomial['vendor_id']=data_polinomial['vendor_id'].astype('category')
data_polinomial['passenger_count']=data_polinomial['passenger_count'].astype('category')


data_polinomial=pd.get_dummies(data_polinomial)


data_train=data_polinomial['trip_duration']
data_polinomial=data_polinomial.drop(['trip_duration'],axis=1)
data_regresion4.dtypes



x_train,x_test,y_train,y_test = train_test_split(data_polinomial,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train



polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)
reg2 = linear_model.LinearRegression()
reg2 = reg2.fit(x_poly, y)
yhat2 = reg2.predict(x_poly)
reg2.coef_

print("Mean squared error: %.2f" % mean_squared_error(y, yhat2))
print('R2: %.2f' % r2_score(y, yhat2))#R2 de 0,64 ha millorat en un 12%
rms = sqrt(mean_squared_error(y, yhat2))#rms de 437.65

x_sorted = x
yhat2_sorted = yhat2
import operator
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,yhat2), key=sort_axis)
x_sorted, yhat2_sorted = zip(*sorted_zip)


plt.plot(x_sorted, yhat2_sorted, color='m')

plt.scatter(y, yhat2, s=10)
plt.show()



# =============================================================================
# Regresion polinomial de nivel 3 no carga
# =============================================================================

data_polinomial2=data_def.copy()

data_polinomial2.drop(['store_and_fwd_flag'
       ,'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)


data_polinomial2['pick_hour']=data_polinomial2['pick_hour'].astype('category')

data_polinomial2['pick_day_week_name']=data_polinomial2['pick_day_week_name'].astype('category')
data_polinomial2['parts_day']=data_polinomial2['parts_day'].astype('category')
data_polinomial2['work_rest_day']=data_polinomial2['work_rest_day'].astype('category')
data_polinomial2['vendor_id']=data_polinomial2['vendor_id'].astype('category')
data_polinomial2['passenger_count']=data_polinomial2['passenger_count'].astype('category')


data_polinomial2=pd.get_dummies(data_polinomial2)


data_train=data_polinomial2['trip_duration']
data_polinomial2.drop(['trip_duration'],axis=1,inplace=True)
data_polinomial2.dtypes



x_train,x_test,y_train,y_test = train_test_split(data_polinomial2,data_train,random_state=10,train_size=0.8)

x=x_train
y=y_train





polynomial_features3 = PolynomialFeatures(degree=4)
x_poly3 = polynomial_features3.fit_transform(x)
reg3 = linear_model.LinearRegression()
reg3 = reg3.fit(x_poly3, y)
yhat3 = reg3.predict(x_poly3)
reg3.coef_

#reordeno para pintar linea de ajuste
x_sorted3 = x
yhat3_sorted = yhat3
import operator
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,yhat3), key=sort_axis)
x_sorted3, yhat3_sorted = zip(*sorted_zip)



print("Mean squared error: %.2f" % mean_squared_error(y, yhat3))
print('R2: %.2f' % r2_score(y, yhat3))
rms = sqrt(mean_squared_error(y, yhat3))#rms de 437.65


#disminuye MSE y aumenta R2

#grafico
plt.scatter(x, y, s=10)
plt.plot(x_sorted3, yhat3_sorted, color='r')

plt.scatter(y, yhat3)

