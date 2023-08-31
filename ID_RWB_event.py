# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:55:40 2023

@author: grant
"""
#%% 
import wavebreakpy as rwb
from netCDF4 import Dataset
import numpy as np
import os
path_py = "C:\Graduate School\RWB Stuff\Python Scripts"
os.chdir(path_py)
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
wavebreak_thres = 15
num_of_overturning = 3
c_round_cont_all = []
c_ext_round_arr_all = []
wb_center_all = []
LC1_all = []
LC2_all = []
LC1_bounds_all= []
LC2_bounds_all = []
LC1_centroids_all = []
LC2_centroids_all = []
matrix_cluster_mean_all = []  
for time_step in np.arange(1,2):
        print(time_step)
        theta_3_worlds = np.concatenate((theta_ex[time_step],theta_ex[time_step],theta_ex[time_step],theta_ex[time_step,:,0].reshape(-1,1)),1)
        theta_3_worlds = np.swapaxes(theta_3_worlds,0,1)
        for isentrope in theta_levels:
        # Call WaveBreak.py 
            c_round_cont, LC1, LC2, LC1_centroids, LC2_centroids, LC1_bounds, LC2_bounds = rwb.WaveBreak(theta_3_worlds,lon_3_worlds,
                                                        isentrope,lat_ext,lon_ext,latlonmesh,num_of_crossings,
                                                        haversine_dist_thres,lat_dist_thres,lon_width_thres)
            
            c_round_cont_all.append(c_round_cont)
            LC1_centroids_all.append(LC1_centroids)
            LC2_centroids_all.append(LC2_centroids)
            # c_ext_round_arr_all.append(c_ext_round_arr)
            # wb_center_all.append(wb_center)
            LC1_all.append(LC1)
            LC2_all.append(LC2)
            LC1_bounds_all.append(LC1_bounds)
            LC2_bounds_all.append(LC2_bounds)

#%%         Limit the dataset to only include overturning contours from the middle domain (360 to 720) 
            # and determine the haversine distance between isentropic levels
        event_centroids_mid = []
        event_bounds_mid = []
        isentrope_dist = []
        RWB_event = []
        matrix_cluster_mean = []
        isentrope_dist_all = []
        possible_overturning_region_idx_all = []
        for isentrope_c, centroids in enumerate(LC1_centroids_all):
            # Only analyze isentropes that have been found to be overturning
            if len(centroids) > 0:
                for cen_idx,lat_lons in enumerate(centroids):
                    if np.logical_and(lat_lons[1] >=360, lat_lons[1] <720):
                        event_centroids_mid.append([lat_lons[0],lat_lons[1],theta_levels[isentrope_c]])  
                        event_bounds_mid.append(LC1_bounds_all[isentrope_c][cen_idx])
        event_centroids_mid = np.stack(event_centroids_mid)
        event_bounds_mid = np.stack(event_bounds_mid)
        # Calculate the Haversine distance between identified overturning contours
        for i, overt_cont in enumerate(event_centroids_mid):
                same_isentropes = np.argwhere(event_centroids_mid[:,-1] == overt_cont[-1])
                other_isentropes = np.argwhere(event_centroids_mid[:,-1] != overt_cont[-1])
                isentrope_dist = np.zeros((len(event_centroids_mid)))
                # Only calculate the Haversine distance for isentropes of different values (e.g., 310 K and 315 K distance not 310 K and 310 K distance)
                for idxs in other_isentropes:
                    overt_cont2 = event_centroids_mid[:,:-1][idxs].squeeze()
                    isentrope_dist[idxs] = rwb.haversine(overt_cont[1], overt_cont[0], overt_cont2[1], overt_cont2[0], 'deg')
                for idxs in same_isentropes:
                    if idxs != i:   
                    # Assign a NaN to isentropes of the same value 
                            isentrope_dist[idxs] = np.nan
            # If there is a 0, that is the overturning isentrope in question                 
            # Identify instances where overturning isentropes occur within 15 degrees of each other
                possible_overturning_region_idx = np.argwhere(np.logical_and(isentrope_dist < wavebreak_thres, isentrope_dist != 0)).squeeze()
                # Add in the index of the isentrope that is being examined for overturning
                possible_overturning_region_idx = np.sort(np.append(possible_overturning_region_idx,i))
                isentrope_dist_all.append(isentrope_dist)
                
                # Cluster isentropes withini the specified distance criteria
                
                flag = False
                for i_prior, prior in enumerate(possible_overturning_region_idx_all):
                    overlap = np.intersect1d(possible_overturning_region_idx, prior)
                    # Is there an isentrope that occurs within the distance criteria in multiple locations
                    if len(overlap) > 0:
                        union = np.union1d(prior,possible_overturning_region_idx)
                        # Cluster isentropes into same event 
                        possible_overturning_region_idx_all[i_prior] = union
                        flag = True
                if flag == False:
                    possible_overturning_region_idx_all.append(possible_overturning_region_idx)
   
        # Now remove any instance where there are not at least the number of user-specified overturnings                
        for possible_event in possible_overturning_region_idx_all:
            if len(possible_event) >= num_of_overturning:
                RWB_event_cent = event_centroids_mid[possible_event]
                overturning_region_bounds = event_bounds_mid[possible_event].squeeze()
                RWB_event.append(RWB_event_cent)
                
                # Fix the "three-worlds" bug that appears near the prime meridian
                
                RWB_event_lon_centroids = RWB_event_cent[:,1]
                RWB_event_lon_cent_range = np.max(RWB_event_lon_centroids) - np.min(RWB_event_cent)
                
                if RWB_event_lon_cent_range > 180:
                    RWB_lon_cen_pM_idx = np.argwhere(RWB_event_lon_centroids>=540)
                    # Subtract 360 to account for the issue at the prime meridian 
                    RWB_event_cent[RWB_lon_cen_pM_idx,1] -= 360
                    # Subtract 360 from same indices in overturning bounds array
                    overturning_region_bounds[RWB_lon_cen_pM_idx,2:] -= 360
                # Identify the north, south, west, and east edges of the overturning region
                north_bound = np.max(overturning_region_bounds[:,0])
                south_bound = np.min(overturning_region_bounds[:,1])
                west_bound = np.min(overturning_region_bounds[:,2])
                east_bound = np.max(overturning_region_bounds[:,3])
                matrix_cluster_mean.append([np.mean(RWB_event_cent[:,0]), np.mean(RWB_event_cent[:,1]), np.mean(RWB_event_cent[:,2]),
                                                    north_bound,south_bound,west_bound,east_bound])
                
        # For no RWBs identified 
        if len(matrix_cluster_mean) < 1:
            matrix_cluster_mean_all.append(matrix_cluster_mean)
        # For RWBs that are identified
        else:
            # Convert the list into an array for easier usage
            matrix_cluster_mean_all.append(np.stack(matrix_cluster_mean))
        # Make sure to clear lists for each time step
        LC1_centroids_all = []
        LC2_centroids_all = []
        LC1_bounds_all = []
        LC2_bounds_all = []
