import xarray as xr
import numpy as np
import pandas as pd

#import raw simulation data
#release from 01-01-2020 back to 01-01-2018
release1 = xr.concat(objs=(xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release0.nc'), xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release0_2.nc')), dim='obs')
#from 01-01-2018 back to 01-01-2016
release2 = xr.concat(objs=(xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release1.nc'), xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release1_2.nc')), dim='obs')
#from 01-01-2016 back to 01-01-2015
release3 = xr.concat(objs=(xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release2.nc'), xr.open_dataset(r'E:\Run_09-08_NonLin_Tides\Domburg_2020-01-01_Stokes=0_730release2_2.nc')), dim='obs')

#release1 and release2 can be concatenated directly (same size)
#release3 has not the same amount of particle released (release for only one year)
#so release3 will be concatenated after some processing
data_xarray = xr.concat(objs=(release1, release2), dim='traj')

#we want to use particle trajectories for 730 days after release
runtime = 730

time1_raw = data_xarray['time'].values
time2_raw = release3['time'].values
x_raw = data_xarray['lon'].values
y_raw = data_xarray['lat'].values
x2_raw = release3['lon'].values
y2_raw = release3['lat'].values
#%% make an array of week numbers for every particle observation, for post-processing
time1_df = pd.DataFrame(time1_raw)
time2_df = pd.DataFrame(time2_raw)

week1_raw = np.empty((len(x_raw), 1461))
week2_raw = np.empty((len(x2_raw), 1096))

week1 = np.empty((len(x_raw), int(runtime)))
week2 = np.empty((len(x2_raw), int(runtime)))

for i in range(1461): 
    week1_raw[:,i] = time1_df[i].dt.isocalendar().week.to_numpy(na_value='nan')
    
for i in range(1096): 
    week2_raw[:,i] = time2_df[i].dt.isocalendar().week.to_numpy(na_value='nan')
#%% #ignore observations after 730 days, to make sure you follow every particle for the same time
x1 = np.empty((len(x_raw), int(runtime)))
y1 = np.empty((len(x_raw), int(runtime)))
x2 = np.empty((len(x2_raw), int(runtime)))
y2 = np.empty((len(x2_raw), int(runtime)))

for i in range(len(x_raw)):
    #nan values have to be skipped, since these are released later in first run 
    #and continue to be advected in the re-run
    x1[i] = (x_raw[i, np.isfinite(x_raw[i])==True])[:runtime]
    y1[i] = (y_raw[i, np.isfinite(y_raw[i])==True])[:runtime]
    week1[i] = (week1_raw[i,np.isfinite(week1_raw[i])==True])[:runtime]

for i in range(len(x2_raw)):
    x2[i] = (x2_raw[i, np.isfinite(x2_raw[i])==True])[:runtime]
    y2[i] = (y2_raw[i, np.isfinite(y2_raw[i])==True])[:runtime] 
    week2[i] = (week2_raw[i,np.isfinite(week2_raw[i])==True])[:runtime]
 
#concatenate all trajectories
x = np.concatenate([x1[:,:runtime], x2[:,:runtime]])
y = np.concatenate([y1[:,:runtime], y2[:,:runtime]])
t = np.concatenate([week1[:,:runtime], week2[:,:runtime]])

#save the converted trajectories, these are used in the Bayesian framework
np.save('x.npy', x)
np.save('y.npy', y)
np.save('t.npy', y)

