#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:07:46 2023

@author: gxl5179

"""

# Import necessary library 
import matplotlib.pyplot as plt
import numpy as np
import h5py
from netCDF4 import Dataset
import os
from datetime import datetime
from datetime import timedelta
path_py = "/Users/gxl5179/Desktop/RWB_WORK/python"

os.chdir(path_py)

# Load in MATLAB and Python versions 

mat = h5py.File('../data/WB_climo_reGrid3/WB_climo_reGrid.1979.mat', 'r')

nc = Dataset('../data/WB_climo.1979.nc')

matrix_LC1_cluster_mat = np.transpose(np.array(mat.get('matrix_LC1_cluster_mean')))

matrix_LC1_cluster_py = np.array(nc.variables['matrix_LC1_cluster_mean'])

# Test some statistics between the arrays

RWB_dates = matrix_LC1_cluster_py[:,-1].astype(int)

r,c,t = np.shape(matrix_LC1_cluster_mat)


# Make an array of datetime 

JF = np.arange(datetime(1979,1,1), datetime(1979,3,1), timedelta(days=1)).astype(datetime)
D = np.arange(datetime(1979,12,1), datetime(1980,1,1), timedelta(days=1)).astype(datetime)

RWB_dt = np.append(D,JF)

for i, dates in enumerate(RWB_dates):
    date_str = str(dates)
    RWB_dates[i] = datetime(year = int(date_str[:4]), month = int(date_str[4:6]), 
                            day = int(date_str[6:8]), hour = int(date_str[8:]))

for tsteps in range(0,t):
    matrix_LC1_cluster_day = matrix_LC1_cluster_mat[:,:,tsteps]
    