# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:31:01 2023

@author: grant

"""

#%% 
# Import necessary library 
import numpy as np
from netCDF4 import Dataset
import os
#%% Load data
path_py = "C:\Graduate School\RWB Stuff\Python Scripts"
os.chdir(path_py)
import wavebreakpy as rwb
data = Dataset("../data/ERA5DTfield_1979reGrid.nc")
#%% Create variables from loaded data and create extended domain for WaveBreak.py
lat_theta = np.array(data.variables['lat'])
lon_theta = np.array(data.variables['lon'])
theta_ex = np.array(data.variables['DT_theta'])
t, y, x = np.shape(theta_ex)
lon_3_worlds = np.arange(0,1081.25,1.25)
theta_levels = np.arange(280,371,5)
lat_ext, lon_ext = np.meshgrid(lon_3_worlds,lat_theta)
latlonmesh = np.concatenate((lon_ext, lat_ext)).reshape(2,y*len(lon_3_worlds))
lat_ext, lon_ext = np.meshgrid(lat_theta,lon_3_worlds)
latlonmesh = np.swapaxes(latlonmesh,0,1) # This step is needed for the knn search (n_samples, n_feature)
num_of_crossings = 3 # How many times must a contour cross a meridian to be identified as overturning?
haversine_dist_thres = 1500 # distance in km
lat_dist_thres = 40 # degrees latitude maximum of identified overturning
lon_width_thres = 5 # degrees longitude minimum of identified 
wavebreak_thres = 15 # degrees great circle distance between overturning contours
num_of_overturning = 3 # the amount of isentropes required to be within x degrees great circle distance in region of overturning

c_round_cont_all = []
c_ext_round_arr_all = []
LC1_centroids_all = []
LC2_centroids_all = []
LC1_all = []
LC2_all = []
LC1_bounds_all= []
LC2_bounds_all = []
matrix_cluster_mean_all = []
RWB_event_all = []
for time_step in np.arange(0,10):
        print(time_step)
        theta_3_worlds = np.concatenate((theta_ex[time_step],theta_ex[time_step],theta_ex[time_step],theta_ex[time_step,:,0].reshape(-1,1)),1)
        theta_3_worlds = np.swapaxes(theta_3_worlds,0,1)
        for isentrope in theta_levels:
        # Call WaveBreak.py 
            c_round_cont, LC1, LC2, LC1_centroids,LC2_centroids, LC1_bounds, LC2_bounds = rwb.WaveBreak(theta_3_worlds,lon_3_worlds,
                                                        isentrope,lat_ext,lon_ext,latlonmesh,num_of_crossings,
                                                        haversine_dist_thres,lat_dist_thres,lon_width_thres)
            # Store all the information from the WaveBreak.py function for each isentrope
            c_round_cont_all.append(c_round_cont)

            LC1_all.append(LC1)
            LC2_all.append(LC2)
            LC1_centroids_all.append(LC1_centroids)
            LC2_centroids_all.append(LC2_centroids)
            LC1_bounds_all.append(LC1_bounds)
            LC2_bounds_all.append(LC2_bounds)
            
        matrix_cluster_mean, RWB_event = rwb.RWB_events(LC2_centroids_all,LC2_bounds_all,theta_levels,
                                                        wavebreak_thres, num_of_overturning)
        matrix_cluster_mean_all.append(matrix_cluster_mean)
        RWB_event_all.append(RWB_event)
        # Clear out the lists for each time step 
        LC1_centroids_all = []
        LC2_centroids_all = []
        LC1_bounds_all = []
        LC2_bounds_all = []
