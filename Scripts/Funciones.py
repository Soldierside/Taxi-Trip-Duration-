#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:05:09 2019

@author: soldierside
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import math
from haversine import haversine
import scipy
from math import sin, cos, sqrt, atan2, radians

os.getcwd()

directori='/Users/soldierside/Documents/Taxi Trip Duration/nyc-taxi-trip-duration/Scripts'


def haversine_(lat1, lng1, lat2, lng2):
    #function to calculate haversine distance between two co-ordinates
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)


def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhattan distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b






def lat_lon_converter(lat1, lon1, lat2, lon2, unit):
    """
    ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    """
    try:
        R = 6371
        dlon = radians(lon2) - radians(lon1)
        dlat = radians(lat2) - radians(lat1)
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        if unit == 'm':
            return distance
        elif unit == 'km':
            return distance
    except ValueError:
        return np.nan


def haversine_pick(df, lat2, lng2):
    lat1=df[6]
    lng1=df[5]
    #function to calculate haversine distance between two co-ordinates
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)

def haversine_drop(df, lat2, lng2):
    lat1=df[8]
    lng1=df[7]
    #function to calculate haversine distance between two co-ordinates
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)


def rmsle(y, y_pred):##revisar porque tengo una funcion de rmsle en algoritmos xgboost
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

