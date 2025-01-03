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
# path_py = "C:\Graduate School\RWB Stuff\Python Scripts"
# path_py = "/gpfs/group/mmg62/default/gxl5179/python"
path_py   = "/glade/u/home/glachat/python/scripts/"
os.chdir(path_py)
import wavebreakpy as rwb
years = np.arange(2023,2023+1)

lon_3_worlds         = np.arange(0,1081.25,1.25)
theta_levels         = np.arange(270,370+1,5)
hemisphere           = 'SH'

# ----------- Variables for changing the identification of overturning contours ------------
num_of_crossings     = 3 # How many times must a contour cross a meridian to be identified as overturning?
haversine_dist_thres = 1500 # distance in km - The minimum distance an overturning region must be for a contour to be counted
lon_width_thres      = 5 # degrees longitude minimum of identified overturning region of contour
lat_dist_thres       = 40 # degrees latitude maximum of identified overturning - the start and end of a contour should not exceed this value

# ----------- Variables for changing the identification of Rossby wave breaking events --------
RWB_width_thres      = 60 # degrees longitude maximum of identified wave break domain (west bound to east bound)
wavebreak_thres      = 15 # degrees great circle distance between overturning contours
num_of_overturning   = 3 # the amount of isentropes required to be within x degrees great circle distance in region of overturning

for yr in years:
    # The file name being created
    f_create = f"/glade/derecho/scratch/glachat/WB_climo_ERA5_py/WB_climo.{yr}.SH.nc"
    data = Dataset(f"/glade/derecho/scratch/glachat/ERA5/ERA5DTfield_{yr}_SH.nc")
    #%% Create variables from loaded data and create extended domain for WaveBreak.py
    lat_theta = np.array(data.variables['lat'])
    lon_theta = np.array(data.variables['lon'])
    theta_ex = np.array(data.variables['DT_theta'][:,lat_theta<=-10])
    if hemisphere == 'SH':
        lat_theta = lat_theta[::-1] # Start at the pole and end at the equator if in the Southern Hemisphere
        lat_theta = lat_theta[lat_theta<=-10]
        theta_ex  = theta_ex[:,::-1]
    t, y, x = np.shape(theta_ex)
    utc_date = np.array(data.variables['utc_date'])
    lat_ext, lon_ext = np.meshgrid(lon_3_worlds,lat_theta)
    latlonmesh = np.concatenate((lon_ext, lat_ext)).reshape(2,y*len(lon_3_worlds))
    lat_ext, lon_ext = np.meshgrid(lat_theta,lon_3_worlds)
    latlonmesh = np.swapaxes(latlonmesh,0,1) # This step is needed for the knn search (n_samples, n_feature)
    c_round_cont_all = []
    c_ext_round_arr_all = []
    LC1_centroids_all = []
    LC2_centroids_all = []
    LC1_all = []
    LC2_all = []
    LC1_bounds_all= []
    LC2_bounds_all = []
    matrix_LC1_cluster_mean_all = []
    matrix_LC2_cluster_mean_all = []
    RWB_event_all = []
      
    for time_step in np.arange(0,t):
            utc_date_step = utc_date[time_step]
            theta_3_worlds = np.concatenate((theta_ex[time_step],theta_ex[time_step],theta_ex[time_step],theta_ex[time_step,:,0].reshape(-1,1)),1)
            theta_3_worlds = np.swapaxes(theta_3_worlds,0,1)
            for isentrope in theta_levels:
            # Call WaveBreak.py 
                c_round_cont, LC1, LC2, LC1_centroids,LC2_centroids, LC1_bounds, LC2_bounds = rwb.WaveBreak(theta_3_worlds,lon_3_worlds,
                                                            isentrope,lat_ext,lon_ext,latlonmesh,num_of_crossings,
                                                            haversine_dist_thres,lat_dist_thres,lon_width_thres,hemisphere)
                # Store all the information from the WaveBreak.py function for each isentrope
                c_round_cont_all.append(c_round_cont)
    
                LC1_all.append(LC1)
                LC2_all.append(LC2)
                LC1_centroids_all.append(LC1_centroids)
                LC2_centroids_all.append(LC2_centroids)
                LC1_bounds_all.append(LC1_bounds)
                LC2_bounds_all.append(LC2_bounds)
                
            matrix_LC1_cluster_mean, AWB_event = rwb.RWB_events3(LC1_centroids_all,LC1_bounds_all,theta_levels,
                                                            wavebreak_thres,RWB_width_thres,num_of_overturning,utc_date_step)
            matrix_LC2_cluster_mean, CWB_event = rwb.RWB_events3(LC2_centroids_all,LC2_bounds_all,theta_levels,
                                                            wavebreak_thres,RWB_width_thres,num_of_overturning,utc_date_step)
            # For time steps without any events, do not append a blank array
            if len(matrix_LC1_cluster_mean) >= 1:     
                # Make sure to add the time step of the identified wave break   
                matrix_LC1_cluster_mean_all.append(matrix_LC1_cluster_mean)
                # RWB_event_all.append(RWB_event)
            if  len(matrix_LC2_cluster_mean) >= 1:
                # Make sure to add the time step of the identified wave break   
                matrix_LC2_cluster_mean_all.append(matrix_LC2_cluster_mean)
                # RWB_event_all.append(RWB_event)
            # Clear out the lists for each time step 
            LC1_centroids_all = []
            LC2_centroids_all = []
            LC1_bounds_all = []
            LC2_bounds_all = []
    # Create two variables from the user functions to save to a netCDF file
    matrix_LC1_cluster_mean_arr = np.vstack(matrix_LC1_cluster_mean_all)
    matrix_LC2_cluster_mean_arr = np.vstack(matrix_LC2_cluster_mean_all)        
    #%% Eliminate wave breaks that exceed a certain thermodynamic threshold set by the user
            
    #%% Create a nc file with Rossby Wave Break identification
    event,i = np.shape(matrix_LC1_cluster_mean_arr) 
    # f_create = "C:/Graduate School/RWB Stuff/data/WB_climo.1980.nc"
    # This creates the .nc file 
    try: ds.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    ds = Dataset(f_create, 'w', format = 'NETCDF4')
    ds.title  = 'Rossby Wave Breaking Event Information'
    event_dim = ds.createDimension('time', None) # None allows the dimension to be unlimited
    # Create a variable for each column of the matrix cluster mean matrix
    
    # The naming convention has been changed to AWB/CWB for easier understanding when this becomes open-source
    lat_centroid_AWB = ds.createVariable('lat_centroid_AWB',np.float64,('time',))
    # DT_theta_fieldnc.units     = 'Kelvin'
    # DT_theta_fieldnc.long_name = 'Potential Temperature at the Dynamic Tropopause'
    
    lon_centroid_AWB = ds.createVariable('lon_centroid_AWB',np.float64,('time',))
    isentrope_AWB    = ds.createVariable('isentrope_average_AWB',np.float64,('time',))
    north_bound_AWB  = ds.createVariable('north_bound_AWB',np.float64,('time',))
    south_bound_AWB  = ds.createVariable('south_bound_AWB',np.float64,('time',))
    west_bound_AWB   = ds.createVariable('west_bound_AWB',np.float64,('time',))
    east_bound_AWB   = ds.createVariable('east_bound_AWB',np.float64,('time',))
    date_AWB         = ds.createVariable('date_AWB',np.int32,('time',))
    
    lat_centroid_AWB[:] = np.double(matrix_LC1_cluster_mean_arr[:,0].squeeze())
    lon_centroid_AWB[:] = np.double(matrix_LC1_cluster_mean_arr[:,1].squeeze())
    isentrope_AWB[:]    = np.double(matrix_LC1_cluster_mean_arr[:,2].squeeze())
    north_bound_AWB[:]  = np.double(matrix_LC1_cluster_mean_arr[:,3].squeeze())
    south_bound_AWB[:]  = np.double(matrix_LC1_cluster_mean_arr[:,4].squeeze())
    west_bound_AWB[:]   = np.double(matrix_LC1_cluster_mean_arr[:,5].squeeze())
    east_bound_AWB[:]   = np.double(matrix_LC1_cluster_mean_arr[:,6].squeeze())
    date_AWB[:]         = np.double(matrix_LC1_cluster_mean_arr[:,7].squeeze())
    
    lat_centroid_CWB = ds.createVariable('lat_centroid_CWB',np.float64,('time',))
    lon_centroid_CWB = ds.createVariable('lon_centroid_CWB',np.float64,('time',))
    isentrope_CWB    = ds.createVariable('isentrope_average_CWB',np.float64,('time',))
    north_bound_CWB  = ds.createVariable('north_bound_CWB',np.float64,('time',))
    south_bound_CWB  = ds.createVariable('south_bound_CWB',np.float64,('time',))
    west_bound_CWB   = ds.createVariable('west_bound_CWB',np.float64,('time',))
    east_bound_CWB   = ds.createVariable('east_bound_CWB',np.float64,('time',))
    date_CWB         = ds.createVariable('date_CWB',np.int32,('time',))
    
    lat_centroid_CWB[:] = np.double(matrix_LC2_cluster_mean_arr[:,0].squeeze())
    lon_centroid_CWB[:] = np.double(matrix_LC2_cluster_mean_arr[:,1].squeeze())
    isentrope_CWB[:]    = np.double(matrix_LC2_cluster_mean_arr[:,2].squeeze())
    north_bound_CWB[:]  = np.double(matrix_LC2_cluster_mean_arr[:,3].squeeze())
    south_bound_CWB[:]  = np.double(matrix_LC2_cluster_mean_arr[:,4].squeeze())
    west_bound_CWB[:]   = np.double(matrix_LC2_cluster_mean_arr[:,5].squeeze())
    east_bound_CWB[:]   = np.double(matrix_LC2_cluster_mean_arr[:,6].squeeze())
    date_CWB[:]         = np.double(matrix_LC2_cluster_mean_arr[:,7].squeeze())
    
    # Make sure to close file once it is finished writing!
    ds.close()
