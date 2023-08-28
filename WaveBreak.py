# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:04:13 2023

@author: Kevin Bowley and Grant LaChat

Contact information: kbowley@psu.edu, glachat@psu.edu

"""

# Import necessary library 
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import os
from sklearn.neighbors import NearestNeighbors
'''
Function inputs
# 3 worlds of theta 
# lat_ext, lon_ext
# theta levels
# adjustable northern threshold
# adjustable number of meridian crossings
# adjustable threshold for wavebreak length for degrees latitude and the haversine distance
# adjustable threshold for width of wavebreak (degrees longitude)
'''
#%% Load data
path_py = "C:\Graduate School\RWB Stuff\Python Scripts"
os.chdir(path_py)
import wavebreakpy as rwb

data = Dataset("../data/ERA5DTfield_1979reGrid.nc")
theta_ex = np.array(data.variables['DT_theta'][9])
lat_theta = np.array(data.variables['lat'])
lon_theta = np.array(data.variables['lon'])
theta_3_worlds = np.concatenate((theta_ex,theta_ex,theta_ex,theta_ex[:,0].reshape(-1,1)),1)
theta_3_worlds = np.swapaxes(theta_3_worlds,0,1)
lon_3_worlds = np.arange(0,1081.25,1.25)
theta_levels = np.arange(280,371,5)
lat_ext, lon_ext = np.meshgrid(lon_3_worlds,lat_theta)
y,x = np.shape(lat_ext)
latlonmesh = np.concatenate((lon_ext, lat_ext)).reshape(2,y*len(lon_3_worlds))
lat_ext, lon_ext = np.meshgrid(lat_theta,lon_3_worlds)
latlonmesh = np.swapaxes(latlonmesh,0,1) # This step is needed for the knn search (n_samples, n_feature)
num_of_crossings = 3 # How many times must a contour cross a meridian to be identified as overturning?
haversine_dist_thres = 1500 # distance in km
lat_dist_thres = 40 # degrees latitude maximum of identified overturning
lon_width_thres = 5 # degrees longitude minimum of identified overturning
#%% Calling the contour function and getting the QuadContour output
plt.figure(dpi = 1000)
cs = plt.contour(lat_ext[:,1:],lon_ext[:,1:],theta_3_worlds[:,1:],[theta_levels[10]])
contour_coord = cs.allsegs[0]
# plt.close()
# The step below will combine all the polygons vertices into one array (identical to the C_round in MATLAB)
#%% Nearest neighbor search for contour vertices on a lat, lon grid
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric = 'euclidean').fit(latlonmesh)
c_round = []
for contour_crd in contour_coord:
    ind = nbrs.kneighbors(contour_crd,return_distance = False)
    c_round.append(latlonmesh[ind].squeeze())
#%% Ensure that the contour extends around the entire hemisphere (no cutoffs)
c_round_cont = []
for c_round_c in c_round:
    # The if statement is used to determine if the contour starts at 0 and ends at 1080 (extends across world)
    if np.logical_and(c_round_c[0,1] == 0,c_round_c[-1,-1] >= 1080):
        c_round_cont.append(c_round_c)
if len(c_round_cont) > 0:
    #%% Step 1: Find the distance (km) between each point of a contour that 
    # extends across the world: domain of three worlds 
    dist_C_ext = []
    for cntr in c_round_cont:
        for idx,cntr_pts in enumerate(cntr):
            # Special condition for the last point to find the distance between the start and end point 
            # of the contour 
            if idx == len(cntr)-1:
                dist_C_ext.append(rwb.haversine(cntr[0][1], cntr[0][0], cntr[-1][1], cntr[-1][0], 'km'))
            else:
                dist_C_ext.append(rwb.haversine(cntr_pts[1], cntr_pts[0], cntr[idx+1][1], cntr[idx+1][0], 'km'))
    # Convert to array for simplicity with indexing in the step below
    dist_C_ext = np.array(dist_C_ext)
    # Find the points along the contour line have no distance between them 
    # (i.e., at the same lat/lon as the prior vertex) and remove them for next step
    c_round_cont_spur = np.array(c_round_cont).squeeze()
    c_round_cont = c_round_cont_spur[np.argwhere(dist_C_ext!=0),:].squeeze()
    # dist_C_ext = dist_C_ext[dist_C_ext != 0].squeeze()
    #%% Step 2: Seek out all regions where we have three endpoints along a single waveguide 
    # which intesect one meridian
# =============================================================================
#     This search identifies: [1]: Looks for at least 3 crossings 
#                             [2]: Removes spurious points from neighboring points falling along same meridian
#                                 This is necessary as the 3 crossings identification has been tripped up by
#                                 spurious points along the same meridian
# =============================================================================
    # This is a very MATLAB-esque way of approaching the issue but could not replicate using Pythonic logic
    shpe = np.shape(c_round_cont)
    c_ext_round = np.empty((shpe))
    c_ext_round[:] = c_round_cont
    zros = np.zeros((shpe))
    c_ext_round = np.hstack((c_ext_round,zros))
    # This is identical to c_ext_round_cont in WaveBreak.m
    c_ext_round = c_ext_round[:,:3]
    # Looks from east to west starting at the highest value and incrementing in the same step as the longitude resolution
    for lons in np.flip(lon_3_worlds):
        lon_ind = np.argwhere(c_round_cont[:,1] == lons)
        counted = []
        # Does the indentified continously extending contour cross any meridian at least 3 times?
        if len(lon_ind) >= num_of_crossings:
            # Find the index of the longitudes in the c_round_c array
            for inds in lon_ind:
                # Ensure that a crossing does not occur at neighboring points (indices)
                # Convert from array to integer for indexing purposes
                inds = int(inds)
                # Allow for the edges to be captured (i.e., 1080 can throw an IndexError)
                if np.logical_and(lons == 1080, inds == len(c_round_cont) - 1):
                    inds = -1
                prior_lon = c_round_cont[inds-1][1]
                current_lon = c_round_cont[inds][1]            
                next_lon = c_round_cont[inds+1][1]
                # Find the starting point of the overturning contour
                if np.logical_and(prior_lon!=current_lon,current_lon!=next_lon):
                    counted.append(inds)
                # Find the end point   
                elif np.logical_and(prior_lon!=current_lon, current_lon == next_lon):
                        end_point1 = inds
                elif np.logical_and(~np.isnan(end_point1), np.logical_and(current_lon==prior_lon,current_lon != next_lon)):
                        end_point2 = inds
                        counted.append(np.round(np.mean((end_point1,end_point2)),0).astype(int))
                        # Reset index for next iteration through the loop 
                        end_point1 = np.nan
                        end_point2 = np.nan 

            if len(counted) >= 3:
                for i, lon_idx in enumerate(counted):
                    if i == len(counted)-2:
                        break
                    else:
                        indxr = counted[i:i+3]
                        lat_dist = np.max(c_round_cont[indxr,0]) - np.min(c_round_cont[indxr,0])
                        haversine_dist = np.sum(dist_C_ext[counted[i]:counted[i+2]+1])
                    # Determine if these overturning points are greater than a user-defined haversine distance minimum 
                    # but no greater than a user-defined latitude span
                    if np.logical_and(haversine_dist>=haversine_dist_thres,lat_dist<=lat_dist_thres):
                        # c_ext_round.append(c_round_cont[counted[i]:counted[i+2] + 1])  
                        c_ext_round[counted[i]:counted[i+2]+1,-1] = 1 
    #%% Step 3: Ensure that the longituinal extent of the WB is at least 5 degrees
    
# =============================================================================
#     
#     Ensure that the overturning exceeds 5 degrees of longitude (width) and ID the wavebreak type
#     
#     This is user-adjustable, so if larger or smaller width wave breaks can be identified
#     
#     Note: MATLAB round and roundn differ from Python's round function 
#     
#     MATLAB round: The round function rounds away from zero to the nearest integer with larger magnitude
#     
#     Python round: Numpy rounds to the nearest even value
#     
#     The below lines of code (id_cont line to the creation of LC1 and LC2 arrays) may appear 
#     and are overly-complicated. However, extensive de-bugging and testing revealed that 
#     having indentified contours as arrays in a list was the easiest way to store information
#     about wave breaking using this function.     
# =============================================================================
    # Make a list of array(s) of identified overturning contours at a given theta level
    id_cont = np.diff(c_ext_round[:,-1])
    # This is very similar/identical to scounter_ext_C and ecounter_ext_C in WaveBreak.m
    id_cont_idx = np.argwhere(id_cont != 0)
    # Every other index is the start of a identified contour from id_cont_idx
    scounter_ext_c = id_cont_idx[0::2]
    ecounter_ext_c = id_cont_idx[1::2]
    c_ext_round_list = []
    for id_cont_counter, scounter in enumerate(scounter_ext_c):
        scounter = int(scounter)
        ecounter = int(ecounter_ext_c[id_cont_counter])
        c_ext_round_list.append(c_ext_round[scounter+1:ecounter+1,:2])
   # Create arrays for LC1 and LC2 events
    LC1 = []
    LC1_centroids = []
    LC1_bounds = []
    # LC1_bounds = np.empty((6,len(c_ext_round)))
    # LC1_bounds[:] = np.nan
   # LC1_cnt = []
    LC2 = []
    LC2_centroids = []
    LC2_bounds = []
    # LC2_bounds = np.empty((6,len(c_ext_round)))
    # LC2_bounds[:] = np.nan
    # LC1_event = 0
    # LC2_event = 0
    # wb_center = np.empty((2,len(c_ext_round)))
    # wb_center[:] = np.nan
    # wb_event = 0
    for i,overturning_cont_coords in enumerate(c_ext_round_list):
        # Turn the list into a numpy array to make indexing easier
        c_ext_round_arr = np.stack(overturning_cont_coords).squeeze()
        # Identify the farthest west longitude value
        lon_min_ovrt = np.min(c_ext_round_arr[:,1])
        # Identify the farthest east longitude value
        lon_max_ovrt = np.max(c_ext_round_arr[:,1])
        
        # Make sure to remove overturning contours not exceeding user-specified longitudinal width
        if lon_max_ovrt - lon_min_ovrt <= lon_width_thres:
            c_ext_round_arr[:] = np.nan
        else:
            # Calculate information about the geographic centroids and bounding region of the overturning contour
            mean_lat = np.mean(c_ext_round_arr[:,0])
            mean_lon = np.mean(c_ext_round_arr[:,1])
            # Farthest north and south latitude values
            north_bound = np.max(c_ext_round_arr[:,0])
            south_bound = np.min(c_ext_round_arr[:,0])
            # Farthest west and east longitude values
            west_bound = np.min(c_ext_round_arr[:,1])
            east_bound = np.max(c_ext_round_arr[:,1])
            # LC1_cnt.append()
            
            # Check orientation of overturning to indentify if it is LC1 (AWB) or LC2 (CWB)
    
            if c_ext_round_arr[0][0] > c_ext_round_arr[-1][0]: # This would AWB since the starting latitude of the contour is farther N
                # Store all of the information about the overturning contour
                LC1.append(c_ext_round_arr)
                LC1_centroids.append([mean_lat,mean_lon])
                # LC1_bounds[0, LC1_event] = mean_lat
                # LC1_bounds[1,LC1_event] = mean_lon
                LC1_bounds.append([north_bound,south_bound,west_bound,east_bound])
                # LC1_bounds[2,LC1_event] = north_bound
                # LC1_bounds[3,LC1_event] = south_bound
                # LC1_bounds[4,LC1_event] = east_bound
                # LC1_bounds[5,LC1_event] = west_bound
                # LC1_event = LC1_event + 1
            elif c_ext_round_arr[0][0] < c_ext_round_arr[-1][0]: # This would be CWB since the starting latitude of the contour is farther S
                # Store all of the information about the overturning contour
                LC2.append(c_ext_round_arr)
                LC2_centroids.append([mean_lat,mean_lon])
                LC2_bounds.append([north_bound,south_bound,west_bound,east_bound])
                # LC2_bounds[0, LC1_event] = mean_lat
                # LC2_bounds[1,LC1_event] = mean_lon
                # LC2_bounds[2,LC1_event] = north_bound
                # LC2_bounds[3,LC1_event] = south_bound
                # LC2_bounds[4,LC1_event] = east_bound
                # LC2_bounds[5,LC1_event] = west_bound
                # LC2_event = LC2_event + 1
            # Store geographic center of all wave break events into seperate array 
            # wb_center[0,wb_event] = mean_lat
            # wb_center[1,wb_event] = mean_lon
            # wb_event = wb_event + 1
# Create blank lists to return from WaveBreak.py if no overturning contours are identified 
else:
    c_round_cont_spur = []
    c_round_cont = []
    LC1 = []
    LC2 = []
    LC1_centroids = []
    LC2_centroids = []
    LC1_bounds = []
    LC2_bounds = []