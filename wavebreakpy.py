
#!/usr/bin/env python3

# Import necessary libraries 

import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from netCDF4 import Dataset

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.neighbors import NearestNeighbors

# A script consisting of all functions used for wavebreaking compositing modified for ERA5 and LENS2 usage

#%%


## FUNCTION ## - The only reason this is here is because another script was not working
def find_coord(centroid_lat,centroid_lon,latdata,londata):
        subtraction_lat = centroid_lat - latdata
        subtraction_lon = centroid_lon - londata
        lat_grdpoint = int(np.argmin(np.abs(subtraction_lat)))
        lon_grdpoint = int(np.argmin(np.abs(subtraction_lon)))
        return lon_grdpoint,lat_grdpoint
## END FUNCTION ##
## FUNCTION ## - 
# %%
def composite(centroid_lat,centroid_lon,latdata,londata,var,season,szn_arr,nsgridpt,wegridpt,latnbound,latsbound):
    sze = np.shape(centroid_lat)
    var_field = []
    # west = []
    # east  = []
    # south  = []
    # north = []
    # center = []
    # Make an array full of NaNs for no wavebreak days
    arr = np.zeros(((wegridpt*2)+1,(nsgridpt*2)+1))
    arr[:] = np.nan
    var_szn = []
    # Build in functionality for seasons 
    if season == "sp" or season == "fa" or season == "su" or season == "wi":
        for idxs in szn_arr:
            var_szn_i = var[:,:,idxs[0]:idxs[1]+1]
            var_szn.append(var_szn_i)
        var_szn_f = np.concatenate(var_szn, axis = 2) 
    else:
        var_szn_f = var
    if len(sze) == 1:
        for idx in range(0,len(centroid_lat)):
            if np.isnan(centroid_lat[idx]):
                var_field.append(arr)
                # north.append(np.nan)
                # south.append(np.nan)
                # west.append(np.nan)
                # east.append(np.nan)
            else:
                # Changes due to data resolution
                center_lon,center_lat = find_coord(centroid_lat[idx],centroid_lon[idx],latdata,londata)
                west_lon = center_lon - wegridpt
                east_lon = center_lon + wegridpt
                north_lat = center_lat + nsgridpt
                # north.append(north_lat)
                south_lat = center_lat - nsgridpt
                # When the data is closer than 25 degrees latitude to either end of the domain
                # if north_lat >= len(latdata):
                #     north_abs = north_lat - len(latdata)
                #     north_lat = latdata[-1]
                #     # south_lat = south_lat - (north_abs+1)
                #     var_1 = var_szn_f[:,south_lat:north_lat+1,:]
                #     pad = np.zeros((41,north_abs,len(centroid_lat)))
                #     pad[:] = np.nan
                #     var_lat = np.concatenate((var_1,pad))
                # elif south_lat < 0:
                #     south_abs = np.abs(south_lat)
                #     var_1 = var_szn_f[:,0:north_lat+1,:]
                #     pad = np.zeros((41,south_abs,len(centroid_lat)))
                #     pad[:] = np.nan
                #     var_lat = np.concatenate((var_1,pad))
                #     # south_lat = 0
                #     # north_lat = north_lat + south_abs
                # else:
                #     var_lat = var_szn_f[:,south_lat:north_lat,:]
                # When the data crosses the prime meridian from either direction
                if west_lon < 0:
                    if north_lat >= len(latdata):
                        north_abs = north_lat - len(latdata)
                        west_abs = np.abs(west_lon)
                        west_lon_idx = len(londata) - west_abs
                        var_1 = var_szn_f[int(west_lon_idx):,int(south_lat):,idx]
                        var_2 = var_szn_f[:int(east_lon)+1,int(south_lat):,idx]
                        pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                        pad[:] = np.nan
                        var_i = np.transpose(np.concatenate((var_1,var_2)))
                        var_f = np.transpose(np.concatenate((var_i,pad)))
                        var_field.append(var_f)
                    elif south_lat < 0:
                        south_abs = np.abs(south_lat)
                        west_abs = np.abs(west_lon)
                        west_lon_idx = len(londata) - west_abs
                        var_1 = var_szn_f[int(west_lon_idx):,:int(north_lat)+1,idx]
                        var_2 = var_szn_f[:int(east_lon)+1,:int(north_lat)+1,idx]
                        pad = np.zeros((south_abs,(wegridpt*2)+1))
                        pad[:] = np.nan
                        var_i = np.transpose(np.concatenate((var_1,var_2)))
                        var_f = np.transpose(np.concatenate((pad,var_i)))
                        var_field.append(var_f)
                    else:
                        west_abs = np.abs(west_lon)
                        west_lon_idx = len(londata) - west_abs
                        var_1 = var_szn_f[int(west_lon_idx):,int(south_lat):int(north_lat)+1,idx]
                        var_2 = var_szn_f[:int(east_lon)+1,int(south_lat):int(north_lat)+1,idx]
                        var_f = np.concatenate((var_1,var_2))
                        var_field.append(var_f)
                elif east_lon >= len(londata):
                    if north_lat >= len(latdata):
                        north_abs = north_lat - len(latdata)
                        north_lat = len(latdata)
                        east_abs = east_lon - len(londata)
                        var_1 = var_szn_f[int(west_lon):,int(south_lat):,idx]
                        var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):,idx]
                        pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                        pad[:] = np.nan
                        var_i = np.transpose(np.concatenate((var_1,var_2)))
                        var_f = np.transpose(np.concatenate((var_i,pad)))
                        var_field.append(var_f)
                    elif south_lat < 0:
                        south_abs = np.abs(south_lat)
                        south_lat = 0
                        east_abs = east_lon - len(londata)
                        var_1 = var_szn_f[int(west_lon):,int(south_lat):int(north_lat)+1,idx]
                        var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx]
                        pad = np.zeros((south_abs,(wegridpt*2)+1))
                        pad[:] = np.nan
                        var_i = np.transpose(np.concatenate((var_1,var_2)))
                        var_f = np.transpose(np.concatenate((pad,var_i)))
                        var_field.append(var_f)
                    else:
                        east_abs = east_lon - len(londata)
                        var_1 = var_szn_f[int(west_lon):,int(south_lat):int(north_lat)+1,idx]
                        var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx]
                        var_f = np.concatenate((var_1,var_2))
                        var_field.append(var_f)
                        
                else:
                        if north_lat >= len(latdata):
                            north_abs = north_lat - len(latdata)
                            var_1 = var_szn_f[int(west_lon):int(east_lon)+1,int(south_lat):,idx]
                            pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                            pad[:] = np.nan
                            var_f = np.transpose(np.concatenate((np.transpose(var_1),pad)))
                            var_field.append(var_f)
                        elif south_lat < 0:
                            south_abs = np.abs(south_lat)
                            var_1 = var_szn_f[int(west_lon):int(east_lon)+1,:int(north_lat)+1,idx]
                            pad = np.zeros((south_abs,(wegridpt*2)+1))
                            pad[:] = np.nan
                            var_f = np.transpose(np.concatenate((pad,np.transpose(var_1))))
                            var_field.append(var_f)
                        else:
                           var_field.append(var_szn_f[int(west_lon):int(east_lon)+1,int(south_lat):int(north_lat)+1,idx])
    else:
        for idx,data in np.ndenumerate(centroid_lat):
            if np.isnan(data):
                pass
                # var_field.append(arr)
                # north.append(np.nan)
                # south.append(np.nan)
                # west.append(np.nan)
                # east.append(np.nan)
            else:
                # Changes due to data resolution
                center_lon,center_lat = find_coord(data,centroid_lon[idx[0],idx[1]],latdata,londata)
                west_lon = center_lon - wegridpt
                east_lon = center_lon + wegridpt
                north_lat = center_lat + nsgridpt
                # north.append(north_lat)
                south_lat = center_lat - nsgridpt
                if data > latnbound or data < latsbound:
                        if IndexError:
                            pass
                            # var_field.append(arr)
                        # var_field.append(arr)
                        pass
                else:
                # When the data is closer than 25 degrees latitude to either end of the domain
                    # if north_lat >= len(latdata):
                    #     north_abs = north_lat - len(latdata)
                    #     north_lat = latdata[-1]
                    #     # south_lat = south_lat - (north_abs+1)
                    #     var_1 = var_szn_f[:,south_lat:north_lat+1,:]
                    #     pad = np.zeros((41,north_abs,len(centroid_lat)))
                    #     pad[:] = np.nan
                    #     var_lat = np.concatenate((var_1,pad))
                    # elif south_lat < 0:
                    #     south_abs = np.abs(south_lat)
                    #     var_1 = var_szn_f[:,0:north_lat+1,:]
                    #     pad = np.zeros((41,south_abs,len(centroid_lat)))
                    #     pad[:] = np.nan
                    #     var_lat = np.concatenate((var_1,pad))
                    #     # south_lat = 0
                    #     # north_lat = north_lat + south_abs
                    # else:
                    var_field.append(var_szn_f[int(west_lon):int(east_lon)+1:,int(south_lat):int(north_lat),idx[0]])
                    # When the data crosses the prime meridian from either direction
                    if west_lon < 0:
                        # if north_lat >= len(latdata):
                        #     north_abs = north_lat - len(latdata)
                        #     west_abs = np.abs(west_lon)
                        #     west_lon_idx = len(londata) - west_abs
                        #     var_1 = var_szn_f[int(west_lon_idx):,int(south_lat):,idx[0]]
                        #     var_2 = var_szn_f[:int(east_lon)+1,int(south_lat):,idx[0]]
                        #     pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                        #     pad[:] = np.nan
                        #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                        #     var_f = np.transpose(np.concatenate((var_i,pad)))
                        #     var_field.append(var_f)
                        # elif south_lat < 0:
                        #     south_abs = np.abs(south_lat)
                        #     west_abs = np.abs(west_lon)
                        #     west_lon_idx = len(londata) - west_abs
                        #     var_1 = var_szn_f[int(west_lon_idx):,:int(north_lat)+1,idx[0]]
                        #     var_2 = var_szn_f[:int(east_lon)+1,:int(north_lat)+1,idx[0]]
                        #     pad = np.zeros((south_abs,(wegridpt*2)+1))
                        #     pad[:] = np.nan
                        #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                        #     var_f = np.transpose(np.concatenate((pad,var_i)))
                        #     var_field.append(var_f)
                        # else:
                        west_abs = np.abs(west_lon)
                        west_lon_idx = len(londata) - west_abs
                        var_1 = var_szn_f[int(west_lon_idx):,int(south_lat):int(north_lat)+1,idx[0]]
                        var_2 = var_szn_f[:int(east_lon)+1,int(south_lat):int(north_lat)+1,idx[0]]
                        var_f = np.concatenate((var_1,var_2))
                        var_field.append(var_f)
                    elif east_lon >= len(londata):
                        # if north_lat >= len(latdata):
                        #     north_abs = north_lat - len(latdata)
                        #     north_lat = len(latdata)
                        #     east_abs = east_lon - len(londata)
                        #     var_1 = var_szn_f[int(west_lon):,int(south_lat):,idx[0]]
                        #     var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):,idx[0]]
                        #     pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                        #     pad[:] = np.nan
                        #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                        #     var_f = np.transpose(np.concatenate((var_i,pad)))
                        #     var_field.append(var_f)
                        # elif south_lat < 0:
                        #     south_abs = np.abs(south_lat)
                        #     south_lat = 0
                        #     east_abs = east_lon - len(londata)
                        #     var_1 = var_szn_f[int(west_lon):,int(south_lat):int(north_lat)+1,idx[0]]
                        #     var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx[0]]
                        #     pad = np.zeros((south_abs,(wegridpt*2)+1))
                        #     pad[:] = np.nan
                        #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                        #     var_f = np.transpose(np.concatenate((pad,var_i)))
                        #     var_field.append(var_f)
                        # else:
                            east_abs = east_lon - len(londata)
                            var_1 = var_szn_f[int(west_lon):,int(south_lat):int(north_lat)+1,idx[0]]
                            var_2 = var_szn_f[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx[0]]
                            var_f = np.concatenate((var_1,var_2))
                            var_field.append(var_f)
                            
                    else:
                            # if north_lat >= len(latdata):
                            #     north_abs = north_lat - len(latdata)
                            #     var_1 = var_szn_f[int(west_lon):int(east_lon)+1,int(south_lat):,idx[0]]
                            #     pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                            #     pad[:] = np.nan
                            #     var_f = np.transpose(np.concatenate((np.transpose(var_1),pad)))
                            #     var_field.append(var_f)
                            # elif south_lat < 0:
                            #     south_abs = np.abs(south_lat)
                            #     var_1 = var_szn_f[int(west_lon):int(east_lon)+1,:int(north_lat)+1,idx[0]]
                            #     pad = np.zeros((south_abs,(wegridpt*2)+1))
                            #     pad[:] = np.nan
                            #     var_f = np.transpose(np.concatenate((pad,np.transpose(var_1))))
                            #     var_field.append(var_f)
                            # else:
                               var_field.append(var_szn_f[int(west_lon):int(east_lon)+1,int(south_lat):int(north_lat)+1,idx[0]])
            # north.append(north_lat)
            # south.append(south_lat)
            # west.append(west_lon)
            # east.append(east_lon)
            # # center.append(center_lat)
    # Now stack the list of arrays
    # composite = np.dstack(var_field)
    # Find the mean lat and lon for plotting
    # north_mean = np.nanmean(north)
    # south_mean = np.nanmean(south)
    # west_mean = np.nanmean(west)
    # east_mean = np.nanmean(east)
    return var_field
   ## END FUNCTION ##    
#%% Updated version of the composite functionn above
# %%
def concatenate_new(centroid_lat,centroid_lon,latdata,londata,var,date,nsgridpt,wegridpt,latnbound,latsbound):
    var_field = []
    event_field = []
    for idx,data in np.ndenumerate(centroid_lat):
        if np.isnan(data):
            continue
        else:
            # Changes due to data resolution
            center_lon,center_lat = find_coord(data,centroid_lon[idx[0],idx[1]],latdata,londata)
            west_lon = center_lon - wegridpt
            east_lon = center_lon + wegridpt
            north_lat = center_lat + nsgridpt
            south_lat = center_lat - nsgridpt
            if np.logical_or(data > latnbound, data < latsbound):
                    # if IndexError:
                    continue
                        # var_field.append(arr)
                    # var_field.append(arr)
            else:
                # When the data crosses the prime meridian from either direction
                if west_lon < 0:
                    # if north_lat >= len(latdata):
                    #     north_abs = north_lat - len(latdata)
                    #     west_abs = np.abs(west_lon)
                    #     west_lon_idx = len(londata) - west_abs
                    #     var_1 = var_szn_f[int(west_lon_idx):,int(south_lat):,idx[0]]
                    #     var_2 = var_szn_f[:int(east_lon)+1,int(south_lat):,idx[0]]
                    #     pad = np.zeros((north_abs+1,(wegridpt*2)+1))
                    #     pad[:] = np.nan
                    #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                    #     var_f = np.transpose(np.concatenate((var_i,pad)))
                    #     var_field.append(var_f)
                    # elif south_lat < 0:
                    #     south_abs = np.abs(south_lat)
                    #     west_abs = np.abs(west_lon)
                    #     west_lon_idx = len(londata) - west_abs
                    #     var_1 = var_szn_f[int(west_lon_idx):,:int(north_lat)+1,idx[0]]
                    #     var_2 = var_szn_f[:int(east_lon)+1,:int(north_lat)+1,idx[0]]
                    #     pad = np.zeros((south_abs,(wegridpt*2)+1))
                    #     pad[:] = np.nan
                    #     var_i = np.transpose(np.concatenate((var_1,var_2)))
                    #     var_f = np.transpose(np.concatenate((pad,var_i)))
                    #     var_field.append(var_f)
                    # else:
                    west_abs = np.abs(west_lon)
                    west_lon_idx = len(londata) - west_abs
                    var_1 = var[int(west_lon_idx):,int(south_lat):int(north_lat)+1,idx[0]]
                    var_2 = var[:int(east_lon)+1,int(south_lat):int(north_lat)+1,idx[0]]
                    var_f = np.concatenate((var_1,var_2))
                    var_field.append(var_f)
                    event_field.append(date[idx[0]])
                elif east_lon >= len(londata):
                        east_abs = east_lon - len(londata)
                        var_1 = var[int(west_lon):,int(south_lat):int(north_lat)+1,idx[0]]
                        var_2 = var[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx[0]]
                        var_f = np.concatenate((var_1,var_2))
                        var_field.append(var_f)
                        event_field.append(date[idx[0]])
                else:
                           var_field.append(var[int(west_lon):int(east_lon)+1,int(south_lat):int(north_lat)+1,idx[0]])
                           event_field.append(date[idx[0]])
            # north.append(north_lat)
            # south.append(south_lat)
            # west.append(west_lon)
            # east.append(east_lon)
            # # center.append(center_lat)
    # Now stack the list of arrays
    composite = np.dstack(var_field)
    # Find the mean lat and lon for plotting
    # north_mean = np.nanmean(north)
    # south_mean = np.nanmean(south)
    # west_mean = np.nanmean(west)
    # east_mean = np.nanmean(east)
    return composite, event_field
   ## END FUNCTION ##    
#%% Function to have create wavebreak event data 
def wbevent2date(lat_centroid,date,nbound,sbound):
    event_date = []
    for idx, data in np.ndenumerate(lat_centroid):
        if np.isnan(data) or np.logical_or(data > nbound, data < sbound):
            continue
        else:
            event_date.append(date[idx[0]])
    return event_date 
#%%
def avg_composite(centroid_lat,centroid_lon,latdata,londata,var):
    x,y,z = np.shape(var)
    var_field = []
    mean_lat_centroid = np.nanmean(centroid_lat)
    mean_lon_centroid = np.nanmean(centroid_lon)
    if mean_lon_centroid>=360 and mean_lon_centroid < 720:
        mean_lon_centroid = mean_lon_centroid - 360
    elif mean_lon_centroid >=720 :
        mean_lon_centroid = mean_lon_centroid - 720
    for idx,data in enumerate(centroid_lat):
        if np.isnan(data) or np.isnan(centroid_lon[idx]):
            pass
        else:
            center_lon,center_lat = find_coord(mean_lat_centroid,mean_lon_centroid,latdata,londata)
            west_lon,south_lat = find_coord(mean_lat_centroid - 25,mean_lon_centroid- 25,latdata,londata) 
            east_lon, north_lat = find_coord(mean_lat_centroid + 25,mean_lon_centroid + 25,latdata,londata)
        var_field.append(var[int(west_lon):int(east_lon)+1,int(south_lat):int(north_lat)+1,idx])
    return var_field,mean_lat_centroid,mean_lon_centroid       
   ## END FUNCTION ##  
#%%
# Function to remove extreme values from composite to avoid messed up location
def rm_extreme(composite):
    extrema = np.argwhere(np.nanmax(composite) == composite)
    for pts in extrema:
        composite[pts[0],pts[1],pts[2], pts[3]] = np.nan
    return composite
#%%
# Function to have the lat/lon centroids organized into a season of choice
def szntrim_ERA5(lat_centroid,lon_centroid,season):
    # Spring
    s_sp = 59 
    e_sp = 150  
    # Summer
    s_su = 151
    e_su = 242 
    # Fall
    s_fa = 243
    e_fa = 333 
    # Winter 
    s_wi = 334
    # List to append all seasons
    lat_centroid_all = []
    lon_centroid_all = []
    szn_arr = []
    # Make an array representing days in a year
    for yr, lat_centroids in enumerate(lat_centroid):
        lon_centroids = lon_centroid[yr]
        if season == "sp":
                     lat_centroid_i = lat_centroids[s_sp:e_sp+1,:]
                     lon_centroid_i = lon_centroids[s_sp:e_sp+1,:]
                     # Need indicies for compositing
                     szn_arr.append((s_sp,e_sp))
                     # Append all entries
                     lat_centroid_all.append(lat_centroid_i)
                     lon_centroid_all.append(lon_centroid_i) 
        elif season == "su":
                    lat_centroid_i = lat_centroids[s_su:e_su+1,:]
                    lon_centroid_i = lon_centroids[s_su:e_su+1,:]
                    szn_arr.append((s_su,e_su))
                    # Append all entries
                    lat_centroid_all.append(lat_centroid_i)
                    lon_centroid_all.append(lon_centroid_i) 
        elif season == "fa":
                     lat_centroid_i = lat_centroids[s_fa:e_fa+1,:]
                     lon_centroid_i = lon_centroids[s_fa:e_fa+1,:]
                     szn_arr.append((s_fa,e_fa))
                    # Append all entries
                     lat_centroid_all.append(lat_centroid_i)
                     lon_centroid_all.append(lon_centroid_i)
        elif season == "wi":
            if yr == 0:
                lat_centroid_i = lat_centroids[s_wi:,:]
                lon_centroid_i = lon_centroids[s_wi:,:]
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i)
                
                del lat_centroid_i
                del lon_centroid_i
                
                lat_centroids = lat_centroid[yr+1]
                lon_centroids = lon_centroid[yr+1]
                
                lat_centroid_i = lat_centroids[:s_sp,:]
                lon_centroid_i = lon_centroids[:s_sp,:]
                
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i)
                
            elif yr == len(lat_centroid) - 1: 
                pass
            else:
                lat_centroids = lat_centroid[yr-1] # Index the December from the previous year first
                lon_centroids = lon_centroid[yr-1]
                lat_centroid_i = lat_centroids[s_wi:,:]
                lon_centroid_i = lon_centroids[s_wi:,:]
                 # Append all entries from December of prior year
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i)
                del lat_centroid_i
                del lon_centroid_i
                # Index the January and February of the current year
                lat_centroids = lat_centroid[yr]
                lon_centroids = lon_centroid[yr]
                lat_centroid_i = lat_centroids[:s_sp,:]
                lon_centroid_i = lon_centroids[:s_sp,:]
                # Append all entries from January and February
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i)
                szn_arr.append((s_wi,365))
                szn_arr.append((0,s_sp))
                    
        else:
            pass
        # Make 1D Arrays out of the mess that I have created
    lat_centroid_arr = np.concatenate(lat_centroid_all)
    lon_centroid_arr = np.concatenate(lon_centroid_all)
           
    return lat_centroid_arr,lon_centroid_arr,szn_arr
#%%
# Function to trim seasons for the atmospheric variable of interest

def sznvartrim_ERA5(atm_var,season):
    # Spring
    s_sp = 59 
    e_sp = 150  
    # Summer
    s_su = 151
    e_su = 242 
    # Fall
    s_fa = 243
    e_fa = 333 
    # Winter 
    s_wi = 334
    # List to append all seasons
    var_all = []
    szn_arr = []
    # Make an array representing days in a year
    for yr, var in enumerate(atm_var):
        if season == "sp":
                     var_i = var[:,:,s_sp:e_sp+1]        
                     # Append all entries
                     var_all.append(var_i)
        elif season == "su":
                    var_i = var[:,:,s_su:e_su+1]
                    szn_arr.append((s_su,e_su))
                    # Append all entries
                    var_all.append(var_i)
        elif season == "fa":
                     var_i = var[:,:,s_fa:e_fa+1]
                    # Append all entries
                     var_all.append(var_i)
        elif season == "wi":
            if yr == 0:
                var_i = var[:,:,s_wi:]
                var_all.append(var_i)
                del var_i
                var = atm_var[yr+1]
                var_i = var[:,:,:s_sp]
                var_all.append(var_i)
            elif yr == len(atm_var) - 1: 
                pass
            else:
                var = atm_var[yr-1] # Index the December from the previous year first
                var_i = var[:,:,s_wi:]
                 # Append all entries from December of prior year
                var_all.append(var_i)
                del var_i
                # Index the January and February of the current year
                var = atm_var[yr]
                var_i = var[:,:,:s_sp]
                # Append all entries from January and February
                var_all.append(var_i)
        else:
            pass
        # Make 1D Arrays out of the mess that I have created
    var_arr = np.concatenate(var_all, axis = 2)
           
    return var_arr
#%%
# Function to trim seasons for the atmospheric variable of interest

def szndatetrim_ERA5(dates_lst,season):
    # Spring
    s_sp = 59 
    e_sp = 150  
    # Summer
    s_su = 151
    e_su = 242 
    # Fall
    s_fa = 243
    e_fa = 333 
    # Winter 
    s_wi = 334
    # List to append all seasons
    var_all = []
    szn_arr = []
    # Make an array representing days in a year
    for yr, var in enumerate(dates_lst):
        if season == "sp":
                     var_i = var[s_sp:e_sp+1]        
                     # Append all entries
                     var_all.append(var_i)
        elif season == "su":
                    var_i = var[s_su:e_su+1]
                    szn_arr.append((s_su,e_su))
                    # Append all entries
                    var_all.append(var_i)
        elif season == "fa":
                     var_i = var[s_fa:e_fa+1]
                    # Append all entries
                     var_all.append(var_i)
        elif season == "wi":
            if yr == 0:
                var_i = var[s_wi:]
                var_all.append(var_i)
                del var_i
                var = dates_lst[yr+1]
                var_i = var[:s_sp]
                var_all.append(var_i)
            elif yr == len(dates_lst) - 1: 
                pass
            else:
                var = dates_lst[yr-1] # Index the December from the previous year first
                var_i = var[s_wi:]
                 # Append all entries from December of prior year
                var_all.append(var_i)
                del var_i
                # Index the January and February of the current year
                var = dates_lst[yr]
                var_i = var[:s_sp]
                # Append all entries from January and February
                var_all.append(var_i)
        else:
            pass
        # Make 1D Arrays out of the mess that I have created
    var_arr = np.concatenate(var_all)
           
    return var_arr
#%%
# Function to have the lat/lon centroids organized into a season of choice
def szntrim(lat_centroid,lon_centroid,season):
    # Spring
    s_sp = 61
    e_sp = 152
    # Summer
    s_su = 153
    e_su = 244
    # Fall
    s_fa = 245
    e_fa = 335
    # Winter 
    s_wi = 336
    e_wi = 425
    # List to append all seasons
    lat_centroid_all = []
    lon_centroid_all = []
    yr_day = np.full((14,1),365)
    szn_arr = []
    # Make an array representing days in a year
    if season == "sp":
             for cal_yr in range(0,len(yr_day)):
                 lat_centroid_i = lat_centroid[s_sp:e_sp+1]
                 lon_centroid_i = lon_centroid[s_sp:e_sp+1]
                 # Need indicies for compositing
                 szn_arr.append((s_sp,e_sp))
                 # Now add a year 
                 s_sp = s_sp + int(yr_day[cal_yr])
                 e_sp = e_sp + int(yr_day[cal_yr])
                 # Append all entries
                 lat_centroid_all.append(lat_centroid_i)
                 lon_centroid_all.append(lon_centroid_i) 
    elif season == "su":
            for cal_yr in range(0,len(yr_day)):
                lat_centroid_i = lat_centroid[s_su:e_su+1]
                lon_centroid_i = lon_centroid[s_su:e_su+1]
                szn_arr.append((s_su,e_su))
                s_su = s_su + int(yr_day[cal_yr])
                e_su = e_su + int(yr_day[cal_yr])
                # Append all entries
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i) 
    elif season == "fa":
            for cal_yr in range(0,len(yr_day)):
                 lat_centroid_i = lat_centroid[s_fa:e_fa+1]
                 lon_centroid_i = lon_centroid[s_fa:e_fa+1]
                 szn_arr.append((s_fa,e_fa))
                 s_fa = s_fa + int(yr_day[cal_yr])
                 e_fa = e_fa + int(yr_day[cal_yr])
                # Append all entries
                 lat_centroid_all.append(lat_centroid_i)
                 lon_centroid_all.append(lon_centroid_i)
    elif season == "wi":
        for cal_yr in range(0,len(yr_day)):
            if cal_yr == 0:
                pass
            else:
                lat_centroid_i = lat_centroid[s_wi:e_wi+1]
                lon_centroid_i = lon_centroid[s_wi:e_wi+1]
                 # Append all entries
                lat_centroid_all.append(lat_centroid_i)
                lon_centroid_all.append(lon_centroid_i)
                szn_arr.append((s_wi,e_wi))
                if cal_yr == 12:
                    break
                else:
                    s_wi = s_wi + int(yr_day[cal_yr])
                    e_wi = e_wi + int(yr_day[cal_yr])
    else:
        pass
    # Make 1D Arrays out of the mess that I have created
    lat_centroid_arr = np.concatenate(lat_centroid_all)
    lon_centroid_arr = np.concatenate(lon_centroid_all)
           
    return lat_centroid_arr,lon_centroid_arr,szn_arr
#%% 
def bybasin(basin,wb_type,lat_centroid,lon_centroid):
    sze = np.shape(lon_centroid)
    if len(sze) == 1:
        for idx,lon in enumerate(lon_centroid):    
            # Basin trims based on criteria (everything is now in degrees east)
            if basin == "Atlantic" and wb_type == "LC1":
                if np.logical_and(lon_centroid[idx] >= 290,lon_centroid[idx] <= 360) or np.logical_and(lon_centroid[idx]>=0,lon_centroid[idx]<=120):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Atlantic" and wb_type == "LC2":
                if np.logical_and(lon_centroid[idx] >= 270,lon_centroid[idx] <= 360) or np.logical_and(lon_centroid[idx]>=0,lon_centroid[idx]<=70):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Pacific" and wb_type == "LC1":
                if np.logical_and(lon_centroid[idx] >= 120,lon_centroid[idx] <= 290):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Pacific" and wb_type == "LC2":
                if np.logical_and(lon_centroid[idx] >= 70,lon_centroid[idx] <= 270):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "None":
                pass
    else:
        for idx,lon in np.ndenumerate(lon_centroid):    
            # Basin trims based on criteria (everything is now in degrees east)
            if basin == "Atlantic" and wb_type == "LC1":
                if np.logical_and(lon >= 290,lon <= 360) or np.logical_and(lon>=0,lon<=120):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Atlantic" and wb_type == "LC2":
                if np.logical_and(lon >= 270,lon_centroid[idx] <= 360) or np.logical_and(lon>=0,lon<=70):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Pacific" and wb_type == "LC1":
                if np.logical_and(lon >= 120,lon<= 290):
                    pass
                else:
                    lon_centroid[idx] = np.nan
                    lat_centroid[idx] = np.nan
                    # del_idx.append(idx)
            elif basin == "Pacific" and wb_type == "LC2":
                if np.logical_and(lon >= 70,lon <= 270):
                    pass
                else:
                    lon_centroid[idx] = 0
                    lat_centroid[idx] = 0
                    # del_idx.append(idx)
            elif basin == "None":
                pass
        
    return lat_centroid, lon_centroid 
#%% Custom Colorbar
def thetaCB():
    a = pd.read_excel("../data/DT_cmap.xlsx", sheet_name="DT_cmap_small")
    k = np.array(a)
    DT_cmap = matplotlib.colors.ListedColormap(k,name = "DT_cmap")
    return DT_cmap
#%% Function to only take NH data 
def var_nh(atm_var,lat_nh,lat):
     nh_idx = len(lat) - len(lat_nh)
     atm_var_nh = atm_var[:,nh_idx:,]
     return atm_var_nh
#%% Function to normalize the longitude from the three worlds to one world (between 0 and 359.75)
def lonNormalize(lon_centroid): 
    for idx,lon in enumerate(lon_centroid):            
        if lon>=360 and lon < 720:
            lon_centroid[idx] = lon - 360
        elif lon >= 720 :
           lon_centroid[idx] = lon - 720
        else:
            pass
    return lon_centroid 
# %% Function to normalize the longitude from the three worlds to one world with multiple wave breaks per day
def lonNormalize2(lon_centroid):
    for idx,lons in enumerate(lon_centroid):
        if lons[0] == np.nan:
            pass
        else:
            for y,longs in enumerate(lons):
                if longs>=360 and longs < 720:
                    lon_centroid[idx,y] = longs - 360
                elif longs >= 720:
                    lon_centroid[idx,y] = longs - 720
                else:
                    pass
    return lon_centroid
# %% Function to find the amount of overturning contours as well as the theta of the overturning contours for each RWB
def contourFinder(matrix_LC1_LC2_loc_s_d_f):
    contour_list = []
    shpe = np.shape(matrix_LC1_LC2_loc_s_d_f)
    for days in range (0,shpe[2]):
        data = matrix_LC1_LC2_loc_s_d_f[:,:,days]
        data[data == 0] = np.nan
        contour_list.append(data[:,2])
    contour_arr = np.vstack(contour_list)
    return contour_arr      
# %% Function to convert index of contour to temperature of overturning contour in Kelvin

def contour2theta(LC1_2_OTC):
    potv_step = np.arange(280,370+5,5)
    for idx, data in np.ndenumerate(LC1_2_OTC):
        if np.isnan(data):
            pass
        elif int(data) == 19:
            LC1_2_OTC[idx] = potv_step[-1]
        else:
            LC1_2_OTC[idx] = potv_step[int(data)] 
    return LC1_2_OTC
# %% Function to convert MATLAB structure to array
def struct2arr(mat,var):
    lst = []
    element = mat['sM']
    dat = element[var]
    lst = [x[0] for x in dat]
    return lst
# %% Function for compositing lower tropospheric fields for the SOM figures
# %%
def comp_lowertrop(centroid_lat,centroid_lon,latdata,londata,var,nsgridpt,wegridpt):
    var_field = []
    for idx,data in enumerate(centroid_lat):
        # Changes due to data resolution
        center_lon,center_lat = find_coord(data,centroid_lon[idx],latdata,londata)
        west_lon = center_lon - wegridpt
        east_lon = center_lon + wegridpt
        north_lat = center_lat + nsgridpt
        south_lat = center_lat - nsgridpt
        # When the data crosses the prime meridian from either direction
        if west_lon < 0:
            west_abs = np.abs(west_lon)
            west_lon_idx = len(londata) - west_abs
            var_1 = var[int(west_lon_idx):,int(south_lat):int(north_lat)+1,idx]
            var_2 = var[:int(east_lon)+1,int(south_lat):int(north_lat)+1,idx]
            var_f = np.concatenate((var_1,var_2))
            var_field.append(var_f)
        elif east_lon >= len(londata):
            east_abs = east_lon - len(londata)
            var_1 = var[int(west_lon):,int(south_lat):int(north_lat)+1,idx]
            var_2 = var[:int(east_abs)+1,int(south_lat):int(north_lat)+1,idx]
            var_f = np.concatenate((var_1,var_2))
            var_field.append(var_f)          
        else:
            var_field.append(var[int(west_lon):int(east_lon)+1,int(south_lat):int(north_lat)+1,idx])
    # Now stack the list of arrays
    composite = np.nanmean(np.dstack(var_field),2)
    
    return composite
   ## END FUNCTION ##    
    
#%% Daily mean function (data formatted from CDS makes this a CDS only "issue")
def dailyaverage(dat):
    daily = []
    idx = 0
    rng1 = np.arange(0,len(dat)+4,4)
    for days in rng1:
        if days == rng1[-1]:
            break
        else:
            daily.append(dat[rng1[idx]:rng1[idx+1]])
            idx = idx + 1
    dailymean = []
    for gph in daily:
        dailymean.append(np.nanmean(gph,0))
    return dailymean 
#%% SOM node frequency for all seasons SOM to find how many times a given season appears in SOM
# One for each SOM node
def som_freq(dates_df,szn,SOM_nodes,bymonth): 
    # Make szn a list of three months 
    freq_list = []
    freq_list_mnt = []
    for m in range(1,SOM_nodes+1):
        szn_counter = 0 # This needs to reset each time to find the frequency in each of the 12 SOM nodes
        month1_counter = 0
        month2_counter = 0
        month3_counter = 0
        node_dates = dates_df[m]
        for i,data in enumerate(node_dates):
            date_str = str(data)
    # Determine if the date occurs during DJF -  to find DJF frequency
            if np.logical_or(date_str[4:6] == szn[0],date_str[4:6] == szn[1]) or date_str[4:6] == szn[2]:
                szn_counter = szn_counter  + 1 
                if bymonth == 1:
                    if date_str[4:6] == szn[0]:
                        month1_counter = month1_counter + 1
                    elif date_str[4:6] == szn[1]:
                        month2_counter = month2_counter + 1
                    elif date_str[4:6] ==  szn[2]:
                        month3_counter = month3_counter + 1
            elif np.isnan(data):
                break
        freq_list.append(((szn_counter/i)*100))
        freq_list_mnt.append((((month1_counter/i)*100),((month2_counter/i)*100),((month3_counter/i)*100)))
        del szn_counter # Delete this variable so that it doesn't influence calculations of other SOM node frequencies
        del month1_counter
        del month2_counter
        del month3_counter
    return freq_list,freq_list_mnt
#%% 
def RWBevent2var(var, lat_centroids_node, lon_centroids_node, lat, lon, wegridpt, nsgridpt):

    var_field = []    
    longrid_all = []
    center_lon,center_lat = find_coord(lat_centroids_node,lon_centroids_node,lat,lon)

    # Create the 50 by 60 degree centered composite

    west_lon = center_lon - wegridpt
    east_lon = center_lon + wegridpt
    north_lat = center_lat + nsgridpt
    south_lat = center_lat - nsgridpt

    
    # When the data crosses the prime meridian from either direction
    if west_lon < 0:
        west_abs = np.abs(west_lon)
        west_lon_idx = len(lon) - west_abs
        var_1 = var[int(south_lat):int(north_lat)+1, int(west_lon_idx):]
        var_2 = var[int(south_lat):int(north_lat)+1, :int(east_lon)+1]
        var_f = np.concatenate((var_1,var_2), axis = 1)
        zonal_mean = np.nanmean(var_f,1)
        merid_mean = np.nanmean(var_f,0)
        latgrid, longrid = np.meshgrid(merid_mean, zonal_mean)
        #var_field.append(var_f)
        var_f_zm = var_f - longrid
        var_field.append(var_f_zm)
        longrid_all.append(longrid)
    elif east_lon >= len(lon):
        east_abs = east_lon - len(lon)
        var_1 = var[int(south_lat):int(north_lat)+1,int(west_lon):]
        var_2 = var[int(south_lat):int(north_lat)+1, :int(east_abs)+1]
        var_f = np.concatenate((var_1,var_2), axis = 1)
        zonal_mean = np.nanmean(var_f,1)
        merid_mean = np.nanmean(var_f,0)
        latgrid, longrid = np.meshgrid(merid_mean, zonal_mean)
        #var_field.append(var_f)
        var_f_zm = var_f - longrid
        var_field.append(np.flip(var_f_zm,0))
        longrid_all.append(longrid)
    else:
        var = var[int(south_lat):int(north_lat)+1, int(west_lon):int(east_lon)+1]
        zonal_mean = np.nanmean(var,1)
        merid_mean = np.nanmean(var,0)
        latgrid, longrid = np.meshgrid(merid_mean, zonal_mean)
        #var_field.append(var)
        var_zm = var - longrid
        var_field.append(np.flip(var_zm,0))
        longrid_all.append(longrid)

    z_m_all = np.dstack(longrid_all)
    zonal_mean = np.nanmean(z_m_all,2)
    return var_field, zonal_mean
#%% Function that backs out the lower troposphere variables of interest
def varSel(var_int,var_int2,lat_centroids_node,lon_centroids_node,multvar,date_str, day, idx, wegridpt, nsgridpt, regrid):

    if np.logical_and(multvar == True,np.logical_and(var_int[9] == "u",var_int2 == None)):
            data = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+"u"+var_int[-3:]+".nc")
            data2 = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+"v"+var_int[-3:]+".nc")
            lat = np.array(data.variables['latitude'])
            var = np.squeeze(np.array(data.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
            var2 = np.squeeze(np.array(data2.variables[var_int[10].upper()][day-1,:,lat>=0,:]))
            # Convert to wind speed
            var = np.sqrt(np.square(var)+np.square(var2))
    elif np.logical_and(multvar == False, np.logical_and(var_int2 == "vec", var_int[9:] == "iwvt")):
            data_iwvt = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5"+var_int[9:]+"field"+date_str[0:6]+".nc")
            lat = np.array(data_iwvt.variables['latitude'])
            var_iwve = np.squeeze(np.array(data_iwvt.variables["VIWVE"][day-1,:,:]))
            var_iwvn = np.squeeze(np.array(data_iwvt.variables["VIWVN"][day-1,:,:]))
            lon = np.array(data_iwvt.variables['longitude']) 
            var_1_evt = RWBevent2var(np.flip(var_iwve,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            var_2_evt = RWBevent2var(np.flip(var_iwvn,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
    elif np.logical_and(multvar == True, np.logical_and(var_int[9] == 'z', var_int2[9] == 'u')):

            data = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+".nc")
            datau = Dataset("/glade/work/jackstone/Data/"+var_int2+"/"+date_str[0:6]+"u"+var_int2[-3:]+".nc")
            datav = Dataset("/glade/work/jackstone/Data/"+var_int2+"/"+date_str[0:6]+"v"+var_int2[-3:]+".nc")
            lat = np.array(data.variables['latitude'])
            var_z = np.squeeze(np.array(data.variables[var_letter][day-1,:,lat>=0,:]))
            var_u = np.squeeze(np.array(datau.variables[var_int2[9].upper()][day-1,:,lat>=0,:]))
            var_v = np.squeeze(np.array(datav.variables[var_int2[10].upper()][day-1,:,lat>=0,:]))
            # Convert to wind speed
            var_uv = np.sqrt(np.square(var_u)+np.square(var_v))
            lon = np.array(data.variables['longitude']) 
            var_1_evt = RWBevent2var(var_z,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            var_2_evt = RWBevent2var(var_uv,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
    elif np.logical_and(multvar == True, var_int2[9:] == 'mslp'):
                
            data_mslp = Dataset("/glade/scratch/jackstone/DATA_IN_SCRATCH_DO_NOT_DELETE/composite_vars/allyears_djf_mslp.nc")
            data_500 = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+".nc")
            data_1000 = Dataset("/glade/work/jackstone/Data/"+var_int[:10]+"1000"+"/"+date_str[0:6]+".nc")
            lat = np.array(data_mslp.variables['latitude'])
            var_z1 = np.squeeze(np.array(data_500.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
            var_z2 = np.squeeze(np.array(data_1000.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
            thickness = np.subtract(var_z1,var_z2)

            var_p = np.squeeze(np.array(data_mslp.variables[var_int2[9:12].upper()][day2idx(date_str),lat>=0,:]))
            lon = np.array(data_mslp.variables['longitude']) 
            var_1_evt = RWBevent2var(thickness,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            var_2_evt = RWBevent2var(var_p,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            
    elif np.logical_and(multvar == True, var_int2[9:11] == 'ze'):
                
            data_vo = Dataset("/glade/work/glachat/data/"+var_int2+"/ERA5vofield"+date_str[0:6]+".nc")
            data_500 = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+".nc")
            lat = np.array(data_500.variables['latitude'])
            lat_vo = np.array(data_vo.variables['latitude'])
            level = np.array(data_vo.variables['level'])
            var_zeta = np.squeeze(np.array(data_vo.variables['VO'][day-1,level==500,:,:]))
            var_z500 = np.squeeze(np.array(data_500.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
            lon = np.array(data_500.variables['longitude']) 
            var_1_evt = np.squeeze(RWBevent2var(np.flip(var_zeta,0),lat_centroids_node[idx], lon_centroids_node[idx], np.flip(lat_vo), lon, wegridpt, nsgridpt))
            var_2_evt = np.squeeze(RWBevent2var(var_z500,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt))

        
    elif np.logical_and(multvar == False, var_int2[9] == 'w'):
                
            data_w = Dataset("/glade/work/glachat/data/"+var_int2+"/ERA5wfield"+date_str[0:6]+".nc")
            data_700 = Dataset("/glade/work/glachat/data/"+var_int+"/"+date_str[0:6]+".nc")
            lat = np.array(data_700.variables['latitude'])
            var_w = np.squeeze(np.array(data_w.variables[var_int2[9].upper()][day-1,:,:,:]))
            var_z700 = np.squeeze(np.array(data_700.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
            lon = np.array(data_700.variables['longitude']) 
            var_2_evt = RWBevent2var(np.flip(var_w,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            var_1_evt = RWBevent2var(var_z500,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
    elif np.logical_and(multvar == False, var_int[9:] == 'pwat'):

        data_pwat = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5"+var_int[9:]+"field"+date_str[0:6]+".nc")
        data_mslp = Dataset("/glade/work/jackstone/Data/composite_vars/allyears_djf_mslp.nc")
        lat = np.array(data_mslp.variables['latitude'])
        var_pwat = np.squeeze(np.array(data_pwat.variables['TCW'][day-1,:,:]))
        var_mslp = np.squeeze(np.array(data_mslp.variables[var_int2[9:12].upper()][day2idx(date_str),lat>=0,:]))
        lon = np.array(data_mslp.variables['longitude']) 
        var_2_evt = RWBevent2var(np.flip(var_pwat,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        var_1_evt = RWBevent2var(var_mslp,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)

    elif np.logical_and(regrid is True, var_int == "ERA5DTfield_"):
        data = Dataset("/glade/work/glachat/data/DT_field_reGrid/"+var_int+date_str[0:4]+"reGrid.nc")
        lat = np.array(data.variables['lat'])
        utc_d = np.array(data.variables["utc_date"])
        # Convert to array of string
        utc_d = [str(d) for d in utc_d]
        # Find index for time dimension
        time_dim = utc_d.index(date_str)
        var_p = np.squeeze(np.array(data.variables['DT_theta'][time_dim,:,:]))
        var_u = np.squeeze(np.array(data.variables['DT_uwind'][time_dim,:,:]))
        var_v = np.squeeze(np.array(data.variables['DT_vwind'][time_dim,:,:]))
        # Convert to wind speed
        var_uv = np.sqrt(np.square(var_u)+np.square(var_v))
        lon = np.array(data.variables['lon']) 
        if np.isnan(lat_centroids_node[idx]):
            var_2_evt = np.zeros((55,49))
            var_2_evt[:,:] = np.nan
            var_1_evt = np.zeros((55,49))
            var_1_evt[:,:] = np.nan
        else:
            var_1_evt = RWBevent2var(var_p,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
            var_2_evt = RWBevent2var(var_uv,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        
    elif np.logical_and(multvar == True, var_int == "ERA5DTfield"):
        data = Dataset("/glade/work/glachat/data/"+var_int+date_str[0:4]+".nc")
        lat = np.array(data.variables['lat'])
        var_t = np.squeeze(np.array(data.variables['DT_pressure'][day2idxDT(date_str),:,:]))
        var_u = np.squeeze(np.array(data.variables['DT_uwind'][day2idxDT(date_str),:,:]))
        var_v = np.squeeze(np.array(data.variables['DT_vwind'][day2idxDT(date_str),:,:]))
        # Convert to wind speed
        var_uv = np.sqrt(np.square(var_u)+np.square(var_v))
        lon = np.array(data.variables['lon']) 
        var_1_evt = RWBevent2var(var_t,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        var_2_evt = RWBevent2var(var_uv,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
    elif np.logical_and(multvar == True, var_int2[9:11] == 'w7'):
        data_w = Dataset("/glade/work/glachat/data/"+var_int2+"/ERA5wfield"+date_str[0:6]+".nc")
        data_700 = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5z700field"+date_str[0:6]+".nc")
        lat = np.array(data_700.variables['latitude'])
        var_w = np.squeeze(np.array(data_w.variables[var_int2[9].upper()][day-1,:,:,:]))
        var_z700 = np.squeeze(np.array(data_700.variables[var_int[9].upper()][day-1,:,lat>=0,:]))
        lon = np.array(data_700.variables['longitude']) 
        var_2_evt = RWBevent2var(var_w,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        var_1_evt = RWBevent2var(var_z700,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        
    elif np.logical_and(multvar == False, var_int[9:] == "iwvt"):
        data_iwvt = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5"+var_int[9:]+"field"+date_str[0:6]+".nc")
        data_mslp = Dataset("/glade/work/jackstone/Data/composite_vars/allyears_djf_mslp.nc")
        lat = np.array(data_mslp.variables['latitude'])
        var_iwvt = np.squeeze(np.array(data_iwvt.variables["Magnitude of Daily IWVT from 1979-2019 DJF"][day-1,:,:]))
        var_mslp = np.squeeze(np.array(data_mslp.variables[var_int2[9:12].upper()][day2idx(date_str,data_mslp),lat>=0,:]))
        lon = np.array(data_mslp.variables['longitude']) 
        var_2_evt = RWBevent2var(var_iwvt,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        var_1_evt = RWBevent2var(var_mslp,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        
    elif np.logical_and(regrid is True, var_int[8:] == "iwvt"):
        data_iwvt = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5"+var_int[8:]+"field"+date_str[0:6]+".nc")
        data_mslp = Dataset("/glade/work/jackstone/Data/composite_vars/allyears_djf_mslp.nc")
        lat = np.array(data_mslp.variables['latitude'])
        var_iwvt = np.squeeze(np.array(data_iwvt.variables["mag_IWVT"][day2idx(date_str,data_iwvt),:,:]))
        var_mslp = np.squeeze(np.array(data_mslp.variables[var_int2[9:12].upper()][day2idx(date_str,data_mslp),lat>=0,:]))
        lon = np.array(data_mslp.variables['longitude']) 
        lat_iwvt = np.array(data_iwvt.variables['lat'])
        lon_iwvt = np.array(data_iwvt.variables['lon'])
        if np.isnan(lat_centroids_node[idx]):
            var_2_evt = np.zeros((55,49))
            var_2_evt[:,:] = np.nan
            var_1_evt = np.zeros((55,49))
            var_1_evt[:,:] = np.nan
        else:
            var_2_evt = RWBevent2var(var_iwvt,lat_centroids_node[idx], lon_centroids_node[idx], lat_iwvt, lon_iwvt, wegridpt, nsgridpt)
            var_1_evt = RWBevent2var(var_mslp,lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)

    elif np.logical_and(multvar == False, np.logical_and(var_int2 == "vec", var_int[9:] == "iwvt")):
        data_iwvt = Dataset("/glade/work/glachat/data/"+var_int+"/ERA5"+var_int[9:]+"field"+date_str[0:6]+".nc")
        lat = np.array(data_mslp.variables['latitude'])
        var_iwve = np.squeeze(np.array(data_iwvt.variables["VIWVE"][day-1,:,:]))
        var_iwvn = np.squeeze(np.array(data_iwvt.variables["VIWVN"][day-1,:,:]))
        lon = np.array(data_mslp.variables['longitude']) 
        var_1_evt = RWBevent2var(np.flip(var_iwve,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
        var_2_evt = RWBevent2var(np.flip(var_iwvn,0),lat_centroids_node[idx], lon_centroids_node[idx], lat, lon, wegridpt, nsgridpt)
    else:
                
        data = Dataset("/glade/work/jackstone/Data/"+var_int+"/"+date_str[0:6]+".nc")
        lat = np.array(data.variables['latitude'])
        var = np.squeeze(np.array(data.variables[var_letter][day-1,:,lat>=0,:]))
            
            
    return var_1_evt, var_2_evt
    
def day2idx(date_str,data):
    
    #d = Dataset("/glade/work/jackstone/Data/composite_vars/allyears_djf_mslp.nc")
    t = np.array(data.variables['time'])
    dt = datetime(year = int(date_str[0:4]), month = int(date_str[4:6]), day = int(date_str[6:8]))
    sub = dt - datetime(year = 1900, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    delta = sub.total_seconds()/3600 # Convert from seconds elapsed since 1900-01-01 to hours elapsed
    idx_min = np.argmin(np.abs(np.subtract(delta,t)))
    return idx_min 

    
def day2idxDT(date_str):
    
    if regrid is True:
        d = Dataset("/glade/work/glachat/data/DT_field_reGrid/ERA5DTfield_"+date_str[0:4]+"reGrid.nc")
        t = np.array(d.variables['time'])
        dt = datetime(year = int(date_str[0:4]), month = int(date_str[4:6]), day = int(date_str[6:8]), hour = int(date_str[8:]))
        sub = dt - datetime(year = 1900, month = 1, day = 1, hour = 0, minute = 0, second = 0)
        delta = sub.total_seconds()/3600 # Convert from seconds elapsed since 1900-01-01 to hours elapsed
        idx_min = np.argmin(np.abs(np.subtract(delta,t)))
        
    else:
        d = Dataset("/glade/work/glachat/data/DT_field_reGrid/ERA5DTfield"+date_str[0:4]+".nc")
        t = np.array(d.variables['time'])

        dt = datetime(year = int(date_str[0:4]), month = int(date_str[4:6]), day = int(date_str[6:8]))
        sub = dt - datetime(year = 1900, month = 1, day = 1, hour = 0, minute = 0, second = 0)
        delta = sub.total_seconds()/3600 # Convert from seconds elapsed since 1900-01-01 to hours elapsed
        idx_min = np.argmin(np.abs(np.subtract(delta,t)))
    return idx_min  
 #%%       
def node_var_comp(west_lon,east_lon,south_lat,north_lat,var,lon,var_field):
    
    # When the data crosses the prime meridian from either direction
    if west_lon < 0:
        west_abs = np.abs(west_lon)
        west_lon_idx = len(lon) - west_abs
        var_1 = var[int(south_lat):int(north_lat)+1, int(west_lon_idx):]
        var_2 = var[int(south_lat):int(north_lat)+1, :int(east_lon)+1]
        var_f = np.concatenate((var_1,var_2), axis = 1)
        var_field.append(var_f)


    elif east_lon >= len(lon):
            east_abs = east_lon - len(lon)
            var_1 = var[int(south_lat):int(north_lat)+1,int(west_lon):]
            var_2 = var[int(south_lat):int(north_lat)+1, :int(east_abs)+1]
            var_f = np.concatenate((var_1,var_2), axis = 1)
            var_field.append(var_f)
    else:
            var = var[int(south_lat):int(north_lat)+1, int(west_lon):int(east_lon)+1]
            var_field.append(var)
          
    return var_field    
    
#%% Custom Colorbar for precipitation
def precipCB():
    a = pd.read_excel("../data/grant_cmaps.xlsx", sheet_name="precip_6")
    k = np.array(a)
    precip_cmap = matplotlib.colors.ListedColormap(k,name = "precip_cmap")
    return precip_cmap    
#%% Custom colorbar for IWVT
def iwvtCB():
    YlOrRd = mpl.colormaps['YlOrRd']
    cint_1 = np.arange(50,501,50) 
    newcolors = YlOrRd(np.linspace(0, 1, len(cint_1)))
    white = np.array([1, 1, 1, 1])
    newcolors[:1, :] = white
    iwvt_cmap = ListedColormap(newcolors)
    return iwvt_cmap
#%% Custom Colorbar for omega
def omegaCB():
    a = pd.read_excel("../data/grant_cmaps.xlsx", sheet_name="omega_8")
    k = np.array(a)
    omega_cmap = matplotlib.colors.ListedColormap(k,name = "omega_cmap")
    return omega_cmap    
#%% Take the west/east bounds index and convert to lon values
def lonboundidx2lon(lon_bound,lon):
    lon_bound_f = []
    # Make 3 worlds
    lon_3 = np.concatenate((lon,lon,lon))
    for lonidxs in lon_bound:
        lon_bound_f.append(lon_3[int(lonidxs)])
    return np.array(lon_bound_f)
#%%  Take the south/north bound index and convert to lat values

def latboundidx2lat(lat_bound,lat):
    lat_bound_f = []
    for latidxs in lat_bound:
        lat_bound_f.append(lat[int(latidxs)])
    return np.array(lat_bound_f)
#%%
def climo_by_node(west_lon_bound,east_lon_bound,south_lat_bound,north_lat_bound,lat,lon):
    y = len(lat)
    x = len(lon)
    climatology_node = np.zeros((y,x))
    for idx,s_lat_b in enumerate(south_lat_bound):
        if ~np.isnan(s_lat_b):
            xw,ys = find_coord(s_lat_b,west_lon_bound[idx],lat,lon)
            xe,yn = find_coord(north_lat_bound[idx],east_lon_bound[idx],lat,lon)
            
        # Wrap around the prime meridian
        if xe<xw:
            climatology_node[ys:yn+1,xw:] = climatology_node[ys:yn+1,xw:] + 1
            climatology_node[ys:yn+1,:xe+1] = climatology_node[ys:yn+1,:xe+1] + 1
        
        else:
            climatology_node[ys:yn+1,xw:xe+1] = climatology_node[ys:yn+1,xw:xe+1] + 1
        
    return climatology_node
#%%
def haversine(lon1, lat1, lon2, lat2, degorkm):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    r = 6371
    
    if degorkm == 'km':
        rng = c * r
        return rng
    
    elif degorkm == 'deg':
        # Unit circle with radius one
        r = 1
        if a.any() < 0:
            a = 0
        elif a.any() > 1:
            a = 1
    
        rng = np.rad2deg(2*r*np.arctan2(np.sqrt(a),np.sqrt(1-a)))
        return rng
#%%
# Rounding function that is identical to MATLAB roundn
def roundn(x,n):
    rounded_value = np.round(x/(10**n))*(10**n)
    return rounded_value
#%% WaveBreak 
# Calling the contour function and getting the QuadContour output
def WaveBreak(theta_3_worlds,lon_3_worlds,theta_level,lat_ext,lon_ext,latlonmesh,num_of_crossings,haversine_dist_thres,lat_dist_thres,lon_width_thres):
    # Calling the contour function and getting the QuadContour output
    cs = plt.contour(lat_ext[:,1:],lon_ext[:,1:],theta_3_worlds[:,1:],[theta_level])
    contour_coord = cs.allsegs[0]
    plt.close()
    # The step below will combine all the polygons vertices into one array (identical to the C_round in MATLAB)
    # Nearest neighbor search for contour vertices on a lat, lon grid
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric = 'euclidean').fit(latlonmesh)
    c_round = []
    for contour_crd in contour_coord:
        ind = nbrs.kneighbors(contour_crd,return_distance = False)
        c_round.append(latlonmesh[ind].squeeze())
    # Ensure that the contour extends around the entire hemisphere (no cutoffs)
    c_round_cont = []
    for c_round_c in c_round:
        # The if statement is used to determine if the contour starts at 0 and ends at 1080 (extends across world)
        if np.logical_and(c_round_c[0,1] == 0,c_round_c[-1,-1] >= 1080):
            c_round_cont.append(c_round_c)
    if len(c_round_cont) > 0:
        # Step 1: Find the distance (km) between each point of a contour that 
        # extends across the world: domain of three worlds 
        dist_C_ext = []
        for cntr in c_round_cont:
            for idx,cntr_pts in enumerate(cntr):
                # Special condition for the last point to find the distance between the start and end point 
                # of the contour 
                if idx == len(cntr)-1:
                    dist_C_ext.append(haversine(cntr[0][1], cntr[0][0], cntr[-1][1], cntr[-1][0], 'km'))
                else:
                    dist_C_ext.append(haversine(cntr_pts[1], cntr_pts[0], cntr[idx+1][1], cntr[idx+1][0], 'km'))
        # Convert to array for simplicity with indexing in the step below
        dist_C_ext = np.array(dist_C_ext)
        # Find the points along the contour line have no distance between them 
        # (i.e., at the same lat/lon as the prior vertex) and remove them for next step
        c_round_cont_spur = np.array(c_round_cont).squeeze()
        c_round_cont = c_round_cont_spur[np.argwhere(dist_C_ext!=0),:].squeeze()
        # dist_C_ext = dist_C_ext[dist_C_ext != 0].squeeze()
        # Step 2: Seek out all regions where we have three endpoints along a single waveguide 
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
                            counted.append(np.ceil(np.mean((end_point1,end_point2))).astype(int))
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
        # Step 3: Ensure that the longituinal extent of the WB is at least 5 degrees
        
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
        if c_ext_round[0,-1] == 1:
            # Must start it at -1 to account for the +1 in line 1420 below
            c_ext_round[0,-1] = -1
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
            # Must add 2 to ecounter to account for Python not stopping at end index
            if ecounter + 2 >= len(c_ext_round):
                c_ext_round_list.append(c_ext_round[scounter+1:,:2])
            else:
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
        for overturning_cont_coords in c_ext_round_list:
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
        c_ext_round_arr = []
    
    return c_round_cont_spur,LC1, LC2, LC1_centroids, LC2_centroids, LC1_bounds, LC2_bounds
#%% Function to identify overturning contours into regions that are wave breaking event
def RWB_events(LC_centroids_all,LC_bounds_all, theta_levels, wavebreak_thres, RWB_width_thres, num_of_overturning, utc_date_step=None):
   
    # These lists will create the variables of interest for 
    event_centroids_mid = []
    event_bounds_mid = []
    isentrope_dist = []
    RWB_event = []
    matrix_cluster_mean = []
    matrix_cluster_mean_all = []
    isentrope_dist_all = []
    possible_overturning_region_idx_all = []
    for isentrope_c, centroids in enumerate(LC_centroids_all):
        # Only analyze isentropes that have been found to be overturning
        if len(centroids) > 0:
            for cen_idx,lat_lons in enumerate(centroids):
                if np.logical_and(lat_lons[1] >=360, lat_lons[1] <720):
                    event_centroids_mid.append([lat_lons[0],lat_lons[1],theta_levels[isentrope_c]])  
                    event_bounds_mid.append(LC_bounds_all[isentrope_c][cen_idx])
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
                isentrope_dist[idxs] = haversine(overt_cont[1], overt_cont[0], overt_cont2[1], overt_cont2[0], 'deg')
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
            # Ensure that overturning region does not exceed 60 degrees longitude width
            if east_bound - west_bound > RWB_width_thres:
                # Pass through iteration in loop without appending to matrix
                continue
            
            else:
                RWB_event.append(RWB_event_cent)
                # Build in independence from utc_date_step
                if utc_date_step == None:
                    matrix_cluster_mean.append([np.mean(RWB_event_cent[:,0]), np.mean(RWB_event_cent[:,1]), np.mean(RWB_event_cent[:,2]),
                                                     north_bound,south_bound,west_bound,east_bound])
                else:   
                    matrix_cluster_mean.append([np.mean(RWB_event_cent[:,0]), np.mean(RWB_event_cent[:,1]), np.mean(RWB_event_cent[:,2]),
                                                north_bound,south_bound,west_bound,east_bound, utc_date_step])
    # For no RWBs identified 
    if len(matrix_cluster_mean) < 1:
        matrix_cluster_mean_all.append(matrix_cluster_mean)
    # For RWBs that are identified
    else:
        # Convert the list into an array for easier usage
        matrix_cluster_mean_all.append(np.stack(matrix_cluster_mean))
 
    return np.array(matrix_cluster_mean_all).squeeze(), RWB_event

