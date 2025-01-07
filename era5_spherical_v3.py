# %%
# Import necessary libraries 

import xarray as xr
import numpy as np
import numpy.ma as ma
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta, date
from metpy.units import units
import metpy.calc as mpcalc
from metpy.interpolate import interpolate_to_isosurface
from spharm import Spharmt, getspecindx
import derivative
import scipy
import wavebreakpy as rwb
import Ngl
# Mapping packages
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import calendar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def calcPV(temperature,uwind,vwind,p_arr,lon,lat,interpolation_value,bottom_or_top):
    '''
    Calculate the potential vorticity using the MetPy potential vorticity baroclinic function 
    The function requires Numpy arrays in the following order:
    Note: The level variable is assumed to start at the top of the atmosphere (zeroth index)
    Temperature (time x level x lat x lon)
    zonal (u) component of the wind (time x level x lat x lon)
    meridional (v) component of the wind (time x level x lat x lon)
    p_arr: 1-D Numpy array (level) of the pressure levels of the data 
    lon: Numpy array of the longitude values
    lat: Numpy array of the latitude values
    interpolation_value: The desired level of interpolation (e.g., 2 PVU = 2*10**-6)
    bottom_or_top: Whether a top-down or bottom up linear interpolation to the level of interest is desired 
    (1 is top-down, 0 is bottom-up)
    Returns:
    Potential temperature on the level of interest
    Pressure on the level of interest
    Temperature on the level of interest
    u-wind on the level of interest
    v-wind on the level of interest
    Potential temperature
    Potential vorticity
    '''
    # Grab the shape of the temperature data for creation of fields on the DT
    data_sh        = np.shape(temperature)
    DT_theta       = np.zeros((data_sh[0],data_sh[2],data_sh[3]))
    DT_temperature = np.zeros((data_sh[0],data_sh[2],data_sh[3]))
    DT_pressure    = np.zeros((data_sh[0],data_sh[2],data_sh[3]))
    DT_u_wind      = np.zeros((data_sh[0],data_sh[2],data_sh[3]))
    DT_v_wind      = np.zeros((data_sh[0],data_sh[2],data_sh[3]))
    PV             = np.zeros((data_sh[0],len(p_arr), data_sh[2], data_sh[3]))
    theta_arr      = np.zeros((data_sh[0],len(p_arr), data_sh[2], data_sh[3]))
    # Calculate potential vorticity using MetPy
    P_arr  = rwb.p2nd(p_arr,temperature,3)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon,lat)
    lat_PV = lat*units.Quantity('degrees')
    # Insert a blank axis to make deltas appear as 3D for derivatives to work in MetPy function
    dx     = dx[None, :, :]
    dy     = dy[None, :, :] 
    lat_PV = lat_PV[None,:,None]
    # Complete PV calculation 
    for t in range(0,data_sh[0]):
        T_arr       = temperature[t]
        u_arr       = uwind[t]
        v_arr       = vwind[t]
        theta_arr[t]= mpcalc.potential_temperature(P_arr*units.hPa,units.Quantity(T_arr,'kelvin'))
        PV[t]       = mpcalc.potential_vorticity_baroclinic(theta_arr[t]*units.Quantity('kelvin'),P_arr*units.Quantity('hPa'),
                                                            u_arr*units.Quantity('m/s'),
                                                            v_arr*units.Quantity('m/s'),
                                                            dx=dx,dy=dy,latitude=lat_PV)
        # Interpolate to Dynamic Tropopause (2 PVU surface)
        DT_theta[t]       = interpolate_to_isosurface(np.array(PV[t]), np.array(theta_arr[t]), interp_val, b_or_t_search)
        DT_pressure[t]    = interpolate_to_isosurface(np.array(PV[t]), np.array(P_arr), interp_val, b_or_t_search)
        DT_temperature[t] = interpolate_to_isosurface(np.array(PV[t]), np.array(T_arr), interp_val, b_or_t_search)
        DT_u_wind[t]      = interpolate_to_isosurface(np.array(PV[t]), np.array(u_arr), interp_val, b_or_t_search)
        DT_v_wind[t]      = interpolate_to_isosurface(np.array(PV[t]), np.array(v_arr), interp_val, b_or_t_search)
    return DT_theta,DT_pressure,DT_temperature,DT_u_wind,DT_v_wind,theta_arr,PV

def calcPV_spharm(temperature,uwind,vwind,p_arr,lon,lat,interpolation_value,bottom_or_top,ntrunc):
    
    '''
    Calculate the potential vorticity using the MetPy potential vorticity baroclinic function 
    The function requires Numpy arrays in the following order:
    Note: The level variable is assumed to start at the top of the atmosphere (zeroth index)
    Temperature (time x level x lat x lon)
    zonal (u) component of the wind (time x level x lat x lon)
    meridional (v) component of the wind (time x level x lat x lon)
    p_arr: 1-D Numpy array (level) of the pressure levels of the data 
    lon: Numpy array of the longitude values
    lat: Numpy array of the latitude values
    interpolation_value: The desired level of interpolation (e.g., 2 PVU = 2*10**-6)
    bottom_or_top: Whether a top-down or bottom up linear interpolation to the level of interest is desired 
    (1 is top-down, 0 is bottom-up)
    Returns:
    Potential temperature on the level of interest
    Pressure on the level of interest
    Temperature on the level of interest
    u-wind on the level of interest
    v-wind on the level of interest
    Potential temperature
    Potential vorticity
    '''
def calcPV_spharm(temperature,uwind,vwind,p_arr,lon,lat,interpolation_value,bottom_or_top,ntrunc,lat_select):
    # Grab the shape of the temperature data for creation of fields on the DT
    data_sh        = np.shape(temperature)
    DT_theta       = np.zeros((data_sh[0],int(data_sh[2]/2),data_sh[3]))
    DT_temperature = np.zeros((data_sh[0],int(data_sh[2]/2),data_sh[3]))
    DT_pressure    = np.zeros((data_sh[0],int(data_sh[2]/2),data_sh[3]))
    DT_u_wind      = np.zeros((data_sh[0],int(data_sh[2]/2),data_sh[3]))
    DT_v_wind      = np.zeros((data_sh[0],int(data_sh[2]/2),data_sh[3]))
    PV             = np.zeros((data_sh[0],len(p_arr),data_sh[2], data_sh[3]))
    theta_arr      = np.zeros((data_sh[0],len(p_arr),data_sh[2], data_sh[3]))

    # Calculate potential vorticity without using MetPy
    P_arr    = rwb.p2nd(p_arr,temperature,3)
    dx, dy   = mpcalc.lat_lon_grid_deltas(lon,lat)
    lat_PV   = lat*units.Quantity('degrees')
    # Use latitude to calculate f term at each grid point for PV calculation below
    if lat_select == 'NH':
        lat_ind = lat>=0
    elif lat_select == 'SH':
        lat_ind = lat<= 0
        
    _, lat_g = np.meshgrid(lons*units.degrees,lat[lat_ind]*units.degrees)
    f        = np.array(mpcalc.coriolis_parameter(lat_g))
    # Insert a blank axis to make deltas appear as 3D for derivatives to work in MetPy function
    dx     = dx[None, :, :]
    dy     = dy[None, :, :] 
    lat_PV = lat_PV[None,:,None]
    # Complete PV calculation 
    for t in range(0,data_sh[0]):
        T_arr       = temperature[t]
        u_arr       = uwind[t]
        v_arr       = vwind[t]
        theta_arr[t]= mpcalc.potential_temperature(P_arr*units.hPa,units.Quantity(T_arr,'kelvin'))
        # Calculate the PV 
        # Must switch axes for spherical harmonics
        u_arr            = np.moveaxis(u_arr,0,-1)
        v_arr            = np.moveaxis(v_arr,0,-1)
    # Must do entire globe starting at the northern-most point and ending at the southern-most point
        theta_arr_sp = np.moveaxis(theta_arr[t],0,-1)
    # Spherical harmonics 
        x         = Spharmt(len(lon),len(lat),legfunc='computed')
    
        scoeffst  = x.grdtospec(theta_arr_sp[::-1], ntrunc=ntrunc)
    
        scoeffsu  = x.grdtospec(u_arr[::-1], ntrunc=ntrunc)
    
        scoeffsv  = x.grdtospec(v_arr[::-1], ntrunc=ntrunc)
    
        dtheta_dx_smooth, dtheta_dy_smooth = x.getgrad(scoeffst)
    
        # Convert u, v, and theta from spectral space (flipping lats back to south to north)
    
        u_arr_smooth     = x.spectogrd(scoeffsu)
        u_arr_smooth     = u_arr_smooth[::-1]
        u_arr_smooth     = np.moveaxis(u_arr_smooth,-1,0)
        u_arr_smooth     = u_arr_smooth[:,lat_ind]
    
        v_arr_smooth     = x.spectogrd(scoeffsv)
        v_arr_smooth     = v_arr_smooth[::-1]
        v_arr_smooth     = np.moveaxis(v_arr_smooth,-1,0)
        v_arr_smooth     = v_arr_smooth[:,lat_ind] 
    
        theta_arr_smooth = x.spectogrd(scoeffst)
        theta_arr_smooth = theta_arr_smooth[::-1]
        theta_arr_smooth = np.moveaxis(theta_arr_smooth,-1,0)
        theta_arr_smooth = theta_arr_smooth[:,lat_ind] 
    
        # Calculate relative vorticity using entire globe starting at northern-most point and ending at southern-most point
        vortcoeffs, divcoeffs              = x.getvrtdivspec(u_arr[::-1],v_arr[::-1],ntrunc=ntrunc) 
        zeta_smooth                        = x.spectogrd(vortcoeffs)
    
        # Convert back to original shapes
    
        zeta_smooth                        = zeta_smooth[::-1]
        dtheta_dx_smooth, dtheta_dy_smooth = dtheta_dx_smooth[::-1,:,:], dtheta_dy_smooth[::-1,:,:]
    
        zeta_smooth                        = np.moveaxis(zeta_smooth,-1,0)
        zeta_smooth                        = zeta_smooth[:,lat_ind]
    
        dtheta_dx_smooth                   = np.moveaxis(dtheta_dx_smooth,-1,0)
        dtheta_dx_smooth                   = dtheta_dx_smooth[:,lat_ind]
    
        dtheta_dy_smooth                   = np.moveaxis(dtheta_dy_smooth,-1,0)
        dtheta_dy_smooth                   = dtheta_dy_smooth[:,lat_ind]
    
        P_nh_arr                           = P_arr[:,lat>=0]
        
        du_dp                              = mpcalc.first_derivative(np.array(u_arr_smooth), x = P_nh_arr*100)
    
        dv_dp                              = mpcalc.first_derivative(np.array(v_arr_smooth), x = P_nh_arr*100)
    
        dtheta_dp                          = mpcalc.first_derivative(theta_arr_smooth, x = P_nh_arr*100)
        # Calculate the PV 
        # Term 1
        term1     = du_dp * dtheta_dy_smooth
        term1_pv  = (-g*term1)
    
        # Term 2
        term2     = dv_dp * dtheta_dx_smooth
        term2_pv  = (g*term2) 
    
        # Hemisphere only variables 
        T_h      = T_arr[:,lat_ind,:]

        # Term 3
        eta       = zeta_smooth + f # Absolute vorticity 
        term3     = eta*dtheta_dp
        term3_pv  = (-g*term3) 
    
        # PV 
        PV = (term1_pv-term2_pv)+term3_pv
    
        # Find the 2 PVU surface from the PV calculation and store variables on the 2 PVU surface into NumPy arrays
        DT_theta[t]       = interpolate_to_isosurface(np.array(PV), np.array(theta_arr_smooth), interp_val, b_or_t_search)
        DT_pressure[t]    = interpolate_to_isosurface(np.array(PV), np.array(P_nh_arr), interp_val, b_or_t_search)
        DT_temperature[t] = interpolate_to_isosurface(np.array(PV), np.array(T_h), interp_val, b_or_t_search)
        DT_u_wind[t]      = interpolate_to_isosurface(np.array(PV), np.array(u_arr_smooth), interp_val, b_or_t_search)
        DT_v_wind[t]      = interpolate_to_isosurface(np.array(PV), np.array(v_arr_smooth), interp_val, b_or_t_search)
    return DT_theta,DT_pressure,DT_temperature,DT_u_wind,DT_v_wind,theta_arr,PV 
def ncsave_ERA5(DT_theta,DT_pressure,DT_temperature,DT_u_wind,DT_v_wind,theta_all,PV_all,era5_p_levs,lat_hemi,lons,time_data,utc_date_arr,year):
#%% Save to netCDF file
        f_create = "/glade/derecho/scratch/glachat/ERA5/ERA5DTfield_{}_SH.nc".format(year)
    # This creates the .nc file 
        try: ds.close()  # just to be safe, make sure dataset is not already open.
        except: pass
        ds = Dataset(f_create, 'w', format = 'NETCDF4')
        ds.title  = 'Calculations of Important Variables at the Dynamic Tropopause (2 PVU surface)'
        # This creates the dimensions within the file 
        time_dim  = ds.createDimension('time', None) # None allows the dimension to be unlimited
        # level_dim = ds.createDimension('level',len(era5_p_levs))
        lat_dim   = ds.createDimension('lat',len(lat_hemi)) # This will represent only the NH to conserve size and number specifies the dimension 
        lon_dim   = ds.createDimension('lon',len(lons))

        latsnc = ds.createVariable('lat',np.float64,('lat',))
        latsnc.units = 'degrees north'
        latsnc.long_name = 'latitude'

        lonnc = ds.createVariable('lon',np.float64,('lon',))
        lonnc.units = 'degrees east'
        lonnc.long_name = 'longitude'

        # levelnc = ds.createVariable('level',np.float64,('level',))
        # levelnc.units = 'hPa'
        # levelnc.long_name = 'Pressure levels'

        utc_dates = ds.createVariable('utc_date',np.int32,('time',))
        utc_dates.units = 'Gregorian_year month day hour'
        utc_dates.long_name = 'UTC date yyyy-mm-dd hh:00:00 as yyyymmddhh'

        times = ds.createVariable('time',np.float64,('time',))
        times.units = 'hours since 1900-01-01 00:00:00'
        times.long_names = 'time'

        DT_theta_fieldnc = ds.createVariable('DT_theta',np.float64,('time','lat','lon',))
        DT_theta_fieldnc.units = 'Kelvin'
        DT_theta_fieldnc.long_name = 'Potential Temperature at the Dynamic Tropopause'

        DT_temp_fieldnc = ds.createVariable('DT_temperature',np.float64,('time','lat','lon',))
        DT_temp_fieldnc.units = 'Kelvin'
        DT_temp_fieldnc.long_name = 'Temperature at the Dynamic Tropopause'

        DT_uwind_fieldnc = ds.createVariable('DT_uwind',np.float64,('time','lat','lon',))
        DT_uwind_fieldnc.units = 'm/s'
        DT_uwind_fieldnc.long_name = 'u-component of the wind at the Dynamic Tropopause'

        DT_vwind_fieldnc = ds.createVariable('DT_vwind',np.float64,('time','lat','lon',))
        DT_vwind_fieldnc.units = 'm/s'
        DT_vwind_fieldnc.long_name = 'v-component of the wind at the Dynamic Tropopause'

        DT_pressure_fieldnc = ds.createVariable('DT_pressure',np.float64,('time','lat','lon',))
        DT_pressure_fieldnc.units = 'hPa'
        DT_pressure_fieldnc.long_name = 'Pressure at the Dynamic Tropopause'

        # PV_fieldnc = ds.createVariable('PV',np.float64,('time','level','lat','lon',))
        # PV_fieldnc.units = 'K kg-1 m2 s-1'
        # PV_fieldnc.long_name = 'Potential Vorticity'
    
        # theta_fieldnc = ds.createVariable('theta',np.float64,('time','level','lat','lon',))
        # theta_fieldnc.units = 'K'
        # theta_fieldnc.long_name = 'Potential Temperature'

        latsnc[:]  = lat_hemi
        lonnc[:]   = lons
        # levelnc[:] = era5_p_levs
      
        times[:] = time_data

        utc_dates[:] = utc_date_arr
        # Insert the DT data into the NetCDF file - use numpy double function so that it can be double precision for MATLAB users

        DT_theta_fieldnc[:,:,:]    = np.double(DT_theta)
        DT_temp_fieldnc[:,:,:]     = np.double(DT_temperature)
        DT_uwind_fieldnc[:,:,:]    = np.double(DT_u_wind) 
        DT_vwind_fieldnc[:,:,:]    = np.double(DT_v_wind)
        DT_pressure_fieldnc[:,:,:] = np.double(DT_pressure)
        # PV_fieldnc[:,:,:,:]        = np.double(PV_all)
        # theta_fieldnc[:,:,:,:]     = np.double(theta_all)
        # Make sure to close file once it is finished writing!
        ds.close()
        print("netCDF file created and closed")

# Load in ERA5 data
# Indicate the range of pressure levels of interest

# The following two variables are used when loading in the ERA5 data to only include data on pressure levels
# between 50 hPa and 850 hPa (to help the load times and ensure that we are not capturing near surface wave breaks or top of the atmosphere wave breaks)
p_bottom      = 850 # hPa
p_top         = 50  # hPa
# Indicate whether a "bottom-up" or "top-down" interpolation is desired (We did top-down and I would strongly advise using top-down)
b_or_t_search = 1 # 1 indicates zeroth index to -1 index (top-down if pressure starts at lowest value or top of atmosphere) while 0 indicates -1 index to zeroth index (bottom up if pressure starts at highest value or bottom of atmosphere) - 
# This is the value to interpolate to - we analyze the dynamic tropopause as the 2 PVU surface (which is nearly standard)
interp_val    = -2*10**-6 # This should be negative for the southern hemisphere and is double checked below
g             = 9.81
years         = np.arange(2023,2023+1)
months        = [6,7,8]
ntrunc        = None # This can be used for truncation if the data is noisy - we used 60 (equivalent to removing features of ~2 degrees or less in grid point space)
lat_select    = 'SH' # indicate whether the NH or SH is of interest

# Grab some example CESM data for re-gridding - all CESM files have same grid, so just pull one file
f_path_lens2 = "/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/hour_6/U200/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h2.U200.1980010100-1989123100.nc"
# Grab an example ERA5 data for re-gridding
f_path_era5_ex = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/198001/e5.oper.an.pl.128_130_t.ll025sc.1980010100_1980010123.nc"

# Source of the data
data_lens2 = xr.open_dataset(f_path_lens2)
data_era5_ex = xr.open_dataset(f_path_era5_ex)

regridder = xe.Regridder(data_era5_ex, data_lens2, "conservative")
'''
Lines 342 - 390 are how I load in the ERA5 fields on NCAR's system and re-grid them to the
grid spacing of the LENS2. You probably don't need this, so feel free to skip down to Line 391
'''
for y in years:
    year_of_int = str(y)
    print(year_of_int + " is processing")
    store_theta_by_month_dt = []
    store_theta_by_month    = []
    store_pressure_by_month = []
    store_t_by_month        = []
    store_u_by_month        = []
    store_v_by_month        = []
    store_time_by_month     = []
    utc_date_all            = []
    p_all                   = []
    store_pv_by_month       = []
    theta_all               = []
    for m in months:
        _,ndays = calendar.monthrange(y,m)
        for d in range(1,ndays+1):
      # Grab the ERA5 data
            f_path_era5_t = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/{y}{m:02d}/e5.oper.an.pl.128_130_t.ll025sc.{y}{m:02d}{d:02d}00_{y}{m:02d}{d:02d}23.nc"
            f_path_era5_u = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/{y}{m:02d}/e5.oper.an.pl.128_131_u.ll025uv.{y}{m:02d}{d:02d}00_{y}{m:02d}{d:02d}23.nc" 
            f_path_era5_v = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/{y}{m:02d}/e5.oper.an.pl.128_132_v.ll025uv.{y}{m:02d}{d:02d}00_{y}{m:02d}{d:02d}23.nc" 
            f_path_era5_ym_t = f_path_era5_t.format(y=y, m=m, d=d)
            f_path_era5_ym_u = f_path_era5_u.format(y=y, m=m, d=d)
            f_path_era5_ym_v = f_path_era5_v.format(y=y, m=m, d=d)
            data_era5_t = xr.open_dataset(f_path_era5_ym_t)
            data_era5_u = xr.open_dataset(f_path_era5_ym_u)
            data_era5_v = xr.open_dataset(f_path_era5_ym_v)
            print('--------------')
            print('Begin data trim')
            # Index by the pressure levels of interest
            data_era5_levels = data_era5_t.level
            levels_of_int    = data_era5_levels[np.logical_and(data_era5_levels<=p_bottom,data_era5_levels>=p_top)]
            data_era5_time   = data_era5_t["time.hour"]
            # We are just interested in the same six hour time stamps as the LENS2
            data_era5_t_trim = data_era5_t.sel(time=data_era5_t['time.hour'].isin([0,6,12,18]))
            data_era5_u_trim = data_era5_u.sel(time=data_era5_u['time.hour'].isin([0,6,12,18]))
            data_era5_v_trim = data_era5_v.sel(time=data_era5_v['time.hour'].isin([0,6,12,18]))
            # We are just interested between 850 hPa and 50 hPa    
            data_era5_t_trim = data_era5_t_trim.sel(level=levels_of_int)
            data_era5_u_trim = data_era5_u_trim.sel(level=levels_of_int)
            data_era5_v_trim = data_era5_v_trim.sel(level=levels_of_int)
            print('--------------')
            print('Begin Regrid')
            # Regrid function generation
            # Regrid the ERA5 data
            regrid_era5_t = regridder(data_era5_t_trim)
            regrid_era5_u = regridder(data_era5_u_trim)
            regrid_era5_v = regridder(data_era5_v_trim)
            # Store the lon and lat data
            lons            = regrid_era5_t.lon.values
            lats            = regrid_era5_t.lat.values
            # Get the NH latitudes
            if lat_select == 'NH':
                lats_hsphr = lats[lats>=0]
                if interp_val <= 0:
                    interp_value = -interp_val
            elif lat_select == 'SH':
                lats_hsphr = lats[lats<=0]  
                if interp_val >= 0:
                    interp_value = -interp_val
            # Trim latitudes to the NH
            regrid_era5_t_f = regrid_era5_t
            regrid_era5_u_f = regrid_era5_u
            regrid_era5_v_f = regrid_era5_v
            
            print('--------------')
            print('Begin PV calculation')
            # Calculate variables on the DT
            DT_theta, DT_pressure, DT_temperature, DT_u_wind, DT_v_wind, theta_all, PV_all = calcPV_spharm(regrid_era5_t_f.T.values,regrid_era5_u_f.U.values,
                                                                                            regrid_era5_v_f.V.values,levels_of_int,
                                                                                            lons,lats,interp_val,
                                                                                            b_or_t_search,ntrunc,lat_select)
            store_theta_by_month_dt.append(DT_theta)
            store_pressure_by_month.append(DT_pressure)
            store_t_by_month.append(DT_temperature)
            store_u_by_month.append(DT_u_wind)
            store_v_by_month.append(DT_v_wind)
            utc_date_all.append(data_era5_t.utc_date[::6])
            store_pv_by_month.append(PV_all)
            store_theta_by_month.append(theta_all)
            store_time_by_month.append((((regrid_era5_t_f.time.values - np.datetime64('1900-01-01 00')).astype('timedelta64[h]')).astype(np.float64)))
        # Combine into one array for each variable
            print('--------------')      
    theta_arr_dt = np.concatenate(store_theta_by_month_dt,axis=0)
    pressure_arr = np.concatenate(store_pressure_by_month,axis=0)
    temp_arr     = np.concatenate(store_t_by_month,axis=0)
    uwind_arr    = np.concatenate(store_u_by_month,axis=0)
    vwind_arr    = np.concatenate(store_v_by_month,axis=0)
    time_arr     = np.concatenate(store_time_by_month,axis=0)
    utc_date_arr = np.concatenate(utc_date_all,axis=0)
    pv_arr       = np.concatenate(store_pv_by_month,axis=0)
    theta_arr    = np.concatenate(store_theta_by_month,axis=0)
#%% Save to netCDF file
    ncsave_ERA5(theta_arr_dt,pressure_arr,temp_arr,uwind_arr,vwind_arr,
            theta_arr,pv_arr,levels_of_int,lats_hsphr,lons,time_arr,utc_date_arr,y)