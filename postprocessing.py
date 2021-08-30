import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import timeit
import matplotlib.colors
import csv
import shapefile
import cartopy

start = timeit.default_timer()

#load variables
x = np.load(r'E:\Run_09-08_NonLin_Tides\x.npy')
y = np.load(r'E:\Run_09-08_NonLin_Tides\y.npy')
t = (np.load(r'E:\Run_09-08_NonLin_Tides\t.npy')).astype(int)

releasedays = len(x)//100

#load other data
basepath = r'C:/Users/bramv/OneDrive - Universiteit Utrecht/UU/Jaar 3/BONZ/Datafiles/'

file_landmask = basepath + '/datafile_landMask_297x_375y'
landdata = np.genfromtxt(file_landmask, delimiter=None)

file_coast = basepath + 'datafile_coastMask_297x_375y'
coastdata  = np.genfromtxt(file_coast, delimiter=None)

file_popmatrices = basepath + 'netcdf_populationInputMatrices_thres50_297x_375y.nc'
popmatrix = xr.open_mfdataset(file_popmatrices)
popmatrix_2020 = (popmatrix['pop_input'].values)[4,:,:]
c_prior = popmatrix_2020 * coastdata

f_prior_week = np.load(basepath + 'fishingMatrix_week_20132020.npy')

#set fishing prior on land to zero (original data measures on some lakes, but are not in simulation)
for i in range(len(f_prior_week)):
    (f_prior_week[i])[landdata == 1] = 0
    
fisheryregions = np.load(basepath + 'fisheryregions.npy')
coastalregions = np.load(basepath + 'coastalregions.npy')

#load grid data
current_data = xr.open_mfdataset('C:/Users/bramv/documents/CMEMS/*.nc')
lons = current_data.coords['longitude'].values
lats = current_data.coords['latitude'].values
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)

xbins = np.linspace(-20, 13, 298)
ybins = np.linspace(40, 65, 376)  

#create colormap for plotting river data
num_colors = 9
cmap_r = plt.get_cmap('Greys', num_colors)
cmap_r2 = matplotlib.colors.ListedColormap(['white', 'black'])

stop = timeit.default_timer()
print('Time in cell loading variables: ', stop - start)  
#%% Load riverdata
start = timeit.default_timer()

def riverData():
    riverShapeFile    = basepath + 'Riverdata_2021/Meijer2021_midpoint_emissions/Meijer2021_midpoint_emissions.shp'
    pollutionFile        = basepath + 'Riverdata_2021/Meijer2021_midpoint_emissions.csv'
    dataArray_ID = 1 #column with yearly waste discharged by river
    
    sf = shapefile.Reader(riverShapeFile)
    
    #extract files within NorthSea
    plottingDomain = [-8.3, 5, 47, 57]
    
    rivers = {}
    rivers['longitude'] = np.array([])
    rivers['latitude'] = np.array([])
    rivers['ID'] = np.array([],dtype=int)
    rivers['dataArray'] = np.array([])
    
    for i1 in range(len(sf.shapes())):
        long = sf.shape(i1).points[0][0]
        lat = sf.shape(i1).points[0][1]
        
        if plottingDomain[0] < long <plottingDomain[1] and plottingDomain[2] < lat < plottingDomain[3]:
            rivers['longitude'] = np.append(rivers['longitude'],long)
            rivers['latitude'] = np.append(rivers['latitude'],lat)
            rivers['ID'] = np.append(rivers['ID'],i1)
            
            
    with open(pollutionFile, 'r',encoding='ascii') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        i1 = 0
        for row in filereader:
            if i1 > 0:
                data_ID = i1-1 
                if i1 == 1:
                    dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                    rivers['dataArray'] = dataArray
                else:
                    if data_ID in rivers['ID']:
                        dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                        rivers['dataArray'] = np.vstack([rivers['dataArray'],dataArray])
            i1 += 1
    
    coastIndices = np.where(coastdata == 1)
    assert(np.shape(coastIndices)[0] == 2), "coastMask.data should be an array where the first dimension of the three is empty"
    
    # array containing indices of rivers not belonging to North Sea, which are to be deleted
    deleteEntries = np.array([],dtype=int)
    
    # matrix corresponding to fieldmesh, with per coastal cell the amount of river pollution
    riverInputMatrix = np.zeros(fieldMesh_x.shape)
    
    # for every river
    for i1 in range(len(rivers['longitude'])):   
        lon_river = rivers['longitude'][i1]
        lat_river = rivers['latitude'][i1]
        dist = 1e10
        # check which point is closest
        for i2 in range(np.shape(coastIndices)[1]):
            lon_coast = lons[coastIndices[1][i2]]
            lat_coast = lats[coastIndices[0][i2]]
        
            lat_dist = (lat_river - lat_coast) * 1.11e2
            lon_dist = (lon_river - lon_coast) * 1.11e2 * np.cos(lat_river * np.pi / 180)
            dist_tmp = np.sqrt(np.power(lon_dist, 2) + np.power(lat_dist, 2))
            
            # save closest distance
            if dist_tmp < dist:
                dist = dist_tmp
                lat_ID = coastIndices[0][i2]
                lon_ID = coastIndices[1][i2]
            
        # if distance to closest point > threshold (3*approx cell length), delete entry
        if dist > 3*0.125*1.11e2:
            deleteEntries = np.append(deleteEntries,i1)
        # else: get pollution river, and add to releasematrix
        else:
            # add plastic input as obtained from the dataset
            riverInputMatrix[lat_ID,lon_ID] += rivers['dataArray'][i1,dataArray_ID]

    return riverInputMatrix

r_prior = riverData()

stop = timeit.default_timer()
print('Time in cell loading river data: ', stop - start)  
#%% Thicken landborder, for plotting
def thickenCoast(coastalprobs, thickness):
    def getLandBorder(landMask,lon,lat,val_add): 
        n_lat = landMask.shape[0]
        n_lon = landMask.shape[1]
            
        for i1 in range(n_lat):
            for i2 in range(n_lon):
                
                check_bot = True
                check_top = True
                check_left = True
                check_right = True
                
                # check whether land is located at boundary
                if i1 == 0:
                    check_top = False
                if i1 == n_lat-1:
                    check_bot = False
                if i2 == 0:
                    check_left = False
                if i2 == n_lon-1:
                    check_right = False
                    
                # check whether cell is land, if so look for coast
                if landMask[i1,i2] == 1:
                    
                    if check_top:
                        if (landMask[i1-1,i2] == 0) or (landMask[i1-1,i2] >= 2):
                            landMask[i1,i2] = -1
                    if check_bot:
                        if (landMask[i1+1,i2] == 0) or (landMask[i1+1,i2] >= 2):
                            landMask[i1,i2] = -1
                    if check_left:
                        if (landMask[i1,i2-1] == 0) or (landMask[i1,i2-1] >= 2):
                            landMask[i1,i2] = -1
                    if check_right:
                        if (landMask[i1,i2+1] == 0) or (landMask[i1,i2+1] >= 2):
                            landMask[i1,i2] = -1
        landMask[landMask == -1] = val_add         
        return landMask
    
    landMask = landdata.copy()
    coastMask = coastdata.copy()
    
    landBorder = landMask.copy()
    val_add = 2
    for i1 in range(thickness):
        landBorder = getLandBorder(landBorder,lons,lats,val_add)
        val_add += 1
    
    def closest_index(lat,lon,mask_test):
        distMat = 1e5 * np.ones(fieldMesh_x.shape)
        
        test_indices = np.where(mask_test == 1)
        
        distMat_lon = (lon - fieldMesh_x[test_indices[0],test_indices[1]])*1.11e2*0.63 #find distances coastal element w.r.t. ocean cells. 0.63 comes from lat=51deg (Zeeland)
        distMat_lat = (lat - fieldMesh_y[test_indices[0],test_indices[1]])*1.11e2
    
        distMat[test_indices[0],test_indices[1]] = np.sqrt(np.power(distMat_lon, 2) + np.power(distMat_lat, 2))    
        
        return np.where(distMat == distMat.min())[0][0],np.where(distMat == distMat.min())[1][0]
    
    ### interpolate beaching to closest coastal cell
    hist_beaching_coast = np.zeros(fieldMesh_x.shape)
    for i1 in range(len(lats)):
        for i2 in range(len(lons)):
            
            if coastalprobs[i1,i2] > 0:
                
                i_lat,i_lon = closest_index(lats[i1],lons[i2],coastMask)
                hist_beaching_coast[i_lat,i_lon] +=coastalprobs[i1,i2]
    
     ### go through the landborder defined above with increased width
    hist_beaching_extended = np.zeros(fieldMesh_x.shape)
    indices_border = np.where(landBorder > 1)            
    for i1 in range(len(indices_border[0])):
        lon_ = lons[indices_border[1][i1]]
        lat_ = lats[indices_border[0][i1]]
        i_lat,i_lon = closest_index(lat_,lon_,coastMask)
        
        hist_beaching_extended[indices_border[0][i1],indices_border[1][i1]] += hist_beaching_coast[i_lat,i_lon]
    return hist_beaching_extended
#%% Apply Bayesian framework to study temporal variability in sources
start = timeit.default_timer()

def find_sourceprobs_temp(x, y, lon, lat):
    #fishery_posterior_cells_notnormalized; cells are the grid cells, later we aggregate for bar charts 
    f_post_cells_nn = np.empty((52,5,375,297))
    #normalized
    f_post_cells_n = np.empty((52,5,375,297))
    #coastal
    c_post_cells_nn = np.empty((52,5,375,297))
    c_post_cells_n = np.empty((52,5,375,297))
    #river
    r_post_cells_nn = np.empty((52,5,375,297))
    r_post_cells_n = np.empty((52,5,375,297))
    f_post_list_n = np.empty((5,52,5))
    f_post_list_n_temp = np.empty((5,52))
    c_post_list_n = np.empty((10,52,5))
    c_post_list_n_temp = np.empty((10,52))
    total_p_week = np.zeros(52)
    likelihood_week = np.empty((52,5,375,297))
    #first calculate fishery probabilities using time-dependent prior
    for i in range(len(x)//100):
        #id of first particle released on that day: releaseday*100, since 100 particles are released per day
        id1 = i*100
        #release date of particle (actually the beaching date)
        #-1 since we are using it as an index
        #add week 53 (index 52) releases to week 1 (index 0), as week 53 is not there in every year
        releaseweek = int(t[id1,0]) - 1
        if releaseweek == 52:
            releaseweek = 0
        releaseyear = i//365
        for j in range(len(x.T)):
            #weeks in real time
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[100*i:100*(i+1),j], y[100*i:100*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior, hist_day is the likelihood of all particles released on day j
            #multiplied with the right week (time-depedent prior)
            f_post_cells_nn[releaseweek, releaseyear :, :] += hist_day * f_prior_week[week,:,:] 
    #coastal and river don't have time-dependent prior; so just calculate likelihood per release week
    #loop over particles
    for i in range(len(x)):
        releaseweek = int(t[i,0]) - 1
        if releaseweek == 52:
            releaseweek = 0
        releaseyear = i//36500
        hist_particle = np.histogram2d(x[i,:], y[i,:], bins=[lon, lat])[0]
        hist_particle = hist_particle.T
        likelihood_week[releaseweek, releaseyear,:,:] += hist_particle
    
    for k in range(52):
        for l in range(5):
            #multiply with prior, likelihood depends on release week, prior is time-independent for coastal and river
            c_post_cells_nn[k,l,:,:] = likelihood_week[k,l,:,:] * c_prior
            r_post_cells_nn[k,l,:,:] = likelihood_week[k,l,:,:] * r_prior
            #normalize based on amount of fishing activity experienced by particles released in that week
            total_p_week[k] = np.sum(f_post_cells_nn[k,:,:,:])
    posterior_av = np.mean(total_p_week)
    posterior_rel = total_p_week / posterior_av
    #Normalize to 40% fishery avg over the year, 50% coastal and 10% river.
    for k in range(52):
        for l in range(5):
            f_post_cells_n[k, l,:,:] = 40*posterior_rel[k]*f_post_cells_nn[k, l,:,:]/np.nansum(f_post_cells_nn[k, l,:,:])
            r_post_cells_n[k, l,:,:] = (1/6)*(100-40*posterior_rel[k])*r_post_cells_nn[k, l,:,:]/np.nansum(r_post_cells_nn[k, l,:,:])
            c_post_cells_n[k, l,:,:] = (5/6)*(100-40*posterior_rel[k])*c_post_cells_nn[k, l,:,:]/np.nansum(c_post_cells_nn[k, l,:,:])
    #aggregate the grid cell probabilities per target region, for bar chart plotting
    for k in range(52):
        for l in range(5):
            for i in range(5):
                f_i = f_post_cells_n[k, l,:,:] * fisheryregions[:,:,i]
                f_post_list_n[i,k,l] = np.sum(f_i)
            for i in range(10):
                c_i = c_post_cells_n[k, l,:,:] * coastalregions[:,:,i]
                r_i = r_post_cells_n[k, l,:,:] * coastalregions[:,:,i]
                #sum river and coastal probabilities, since both use the same target regions
                cr_i = c_i + r_i
                c_post_list_n[i,k,l] = np.nansum(cr_i)
    
    #take average over the 5 years
    for k in range(52):
        for i in range(10):
            c_post_list_n_temp[i,k] = np.mean(c_post_list_n[i,k,:])  
        for i in range(5):
            f_post_list_n_temp[i,k] = np.mean(f_post_list_n[i,k,:])
            
    return f_post_list_n_temp, c_post_list_n_temp, f_post_cells_n, c_post_cells_n, r_post_cells_n

f_post_list_n_temp, c_post_list_n_temp, f_post_cells_n_temp, c_post_cells_n_temp, r_post_cells_n_temp = find_sourceprobs_temp(x,y,xbins,ybins)

stop = timeit.default_timer()
print('Time calculating temporal variability: ', stop - start)  
#%% Apply Bayesian framework to study influence of age assumption
start = timeit.default_timer()

def find_sourceprobs_age(x, y, lon, lat):
    f_post_cells_nn = np.empty((24, 375, 297))
    f_post_cells_n = np.empty((24, 375, 297))
    c_post_cells_nn = np.empty((24,375,297))
    c_post_cells_n = np.empty((24,375,297))
    r_post_cells_nn = np.empty((24,375,297))
    r_post_cells_n = np.empty((24,375,297))
    f_post_list_n = np.empty((5,24))
    c_post_list_n = np.empty((10,24))
    total_p_week = np.zeros(24)
    oob_pct = np.zeros(24)   
    start1 = timeit.default_timer()
    #again, first calculate fishery probabilities with time-dependent prior
    for i in range(releasedays):
        id1 = 100*i
        #-10 ugly solution, because 24*30 = 720, but I have 730 observations
        for j in range(len(x.T) - 10):
            #assuming 30 days per month
            age = j//30
            #-1 since you are using indices
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[100*i:100*(i+1),j], y[100*i:100*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior
            f_post_cells_nn[age, :, :] += hist_day * f_prior_week[week,:,:] 
    stop1 = timeit.default_timer()
    print('Time calculating fishery probabilities age: ', stop1 - start1)
    #calculate coastal probabilities with constant prior
    #loop over particle age [months]
    for k in range(24):
        #calculate likelihood per assumed particle age (months)
        likelihood = np.zeros((375,297))
        #only consider part of trajectory with right age
        for i in range(len(x)): 
            hist_particle= np.histogram2d(x[i,30*k:30*(k+1)], y[i,30*k:30*(k+1)], bins=[lon, lat])[0]
            hist_particle = hist_particle.T
            likelihood += hist_particle
        #multiply with prior
        c_post_cells_nn[k,:,:] = likelihood * c_prior
        r_post_cells_nn[k,:,:] = likelihood * r_prior
        
        #check how many particles are out of bounds, located at NaN, NaN (lon,lat)
        #weigh fishing activity experienced by particles with that, to compensate for out-of-bounds behavior
        oob_count = np.isnan(x[:,30*k:30*(k+1)]).sum()
        oob_pct[k] = ((oob_count/(30*len(x)))*100)         
        w = 100 - oob_pct[k]
        #total unnormalized probability for age k
        total_p_week[k] = np.nansum(f_post_cells_nn[k,:,:]) / w
    posterior_av = np.mean(total_p_week)
    posterior_rel = total_p_week / posterior_av
    #again, to normalize to 40% avg over the year
    for k in range(24):
        f_post_cells_n[k] = 40*posterior_rel[k]*f_post_cells_nn[k]/np.nansum(f_post_cells_nn[k])
        r_post_cells_n[k] = (1/6)*(100-40*posterior_rel[k])*r_post_cells_nn[k]/np.nansum(r_post_cells_nn[k])
        c_post_cells_n[k] = (5/6)*(100-40*posterior_rel[k])*c_post_cells_nn[k]/np.nansum(c_post_cells_nn[k])
        for i in range(5):
            f_i = f_post_cells_n[k] * fisheryregions[:,:,i]
            f_post_list_n[i,k] = np.sum(f_i)
        for i in range(10):
            c_i = c_post_cells_n[k] * coastalregions[:,:,i]
            r_i = r_post_cells_n[k] * coastalregions[:,:,i]
            cr_i = c_i + r_i
            c_post_list_n[i,k] = np.nansum(cr_i)
    return f_post_list_n, c_post_list_n, f_post_cells_n, r_post_cells_n, c_post_cells_n

f_post_list_n_age, c_post_list_n_age, f_post_cells_n_age, r_post_cells_n_age, c_post_cells_n_age = find_sourceprobs_age(x,y,xbins,ybins)

stop = timeit.default_timer()
print('Time calculating age variability: ', stop - start)
#%%also calculate source probabilities without making any assumption about age, and averaging over all release dates
#most general view
start = timeit.default_timer()

def find_sourceprobs_avg(x, y, lon, lat):
    f_post_cells_nn = np.empty((375, 297))
    f_post_cells_n = np.empty((375, 297))
    c_post_cells_nn = np.empty((375,297))
    c_post_cells_n = np.empty((375,297))
    r_post_cells_nn = np.empty((375,297))
    r_post_cells_n = np.empty((375,297))
    #again, first calculate fishery probabilities with time-dependent prior
    for i in range(releasedays):
        id1 = 100*i
        for j in range(len(x.T)):
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[100*i:100*(i+1),j], y[100*i:100*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior
            f_post_cells_nn += hist_day * f_prior_week[week,:,:] 
    #calculate probabilities for coastal and river, with constant priors
    likelihood = np.zeros((375,297))
    for i in range(len(x)): 
        hist_particle= np.histogram2d(x[i,:], y[i,:], bins=[lon, lat])[0]
        hist_particle = hist_particle.T
        likelihood += hist_particle
    #multiply with prior
    c_post_cells_nn = likelihood * c_prior
    r_post_cells_nn = likelihood * r_prior
    
    #normalize
    f_post_cells_n = 40*f_post_cells_nn/np.sum(f_post_cells_nn)
    r_post_cells_n = 10*r_post_cells_nn/np.sum(r_post_cells_nn)
    c_post_cells_n = 50*c_post_cells_nn/np.sum(c_post_cells_nn)
    return f_post_cells_n, r_post_cells_n, c_post_cells_n

f_post_cells_n_avg, r_post_cells_n_avg, c_post_cells_n_avg = find_sourceprobs_avg(x,y,xbins,ybins) 

stop = timeit.default_timer()
print('Time calculating avg sources: ', stop - start)
#%% Plotting fig 2
start = timeit.default_timer()

levels_mpw = np.logspace(np.log10(0.001), np.log10(1), 9)
levels_fish = np.logspace(np.log10(0.001), np.log10(1), 9)
levels_river = np.logspace(np.log10(0.01), np.log10(10), 9)
fig,ax = plt.subplots(3)
X,Y = np.meshgrid(np.linspace(0,100,100),np.linspace(0,100,100))
plt1 = ax[0].contourf(X,Y,np.random.choice(levels_mpw,size=[100,100]),levels_mpw,cmap=plt.cm.Reds, norm=plt.cm.colors.LogNorm(), extend='both')
cbar1 = plt.colorbar(plt1)
plt2 = ax[1].contourf(X,Y,np.random.choice(levels_fish,size=[100,100]),levels_fish,cmap=plt.cm.Blues, norm=plt.cm.colors.LogNorm(), extend='both')
cbar2 = plt.colorbar(plt2)
plt.close()
plt3 = ax[2].contourf(X,Y,np.random.choice(levels_river,size=[100,100]),levels_river,cmap=plt.cm.Greys, norm=plt.cm.colors.LogNorm(), extend='max')
cbar3 = plt.colorbar(plt3)
 #thicken coast, for plotting
coastalprobs_total = thickenCoast(c_post_cells_n_avg, 3)

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.LAND)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   color='gray', alpha=0.5, linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False

ax.set_extent((-15, 5, 46, 60), ccrs.PlateCarree())
plt.contourf(fieldMesh_x,fieldMesh_y,coastalprobs_total,levels=levels_mpw,extend='both',cmap=plt.cm.Reds,norm=plt.cm.colors.LogNorm())
plt.contourf(fieldMesh_x,fieldMesh_y,f_post_cells_n_avg,levels=levels_fish,extend='both',cmap=plt.cm.Blues, norm=plt.cm.colors.LogNorm())
#dont plot river probabilities below 0.01%, to prevent plot from being too crowded
r_post_cells_n_avg[ r_post_cells_n_avg < 1e-2 ] = 'nan'

for i in range(len(lats)):
    im = ax.scatter(fieldMesh_x[i],fieldMesh_y[i], c = r_post_cells_n_avg[i], cmap=cmap_r, vmin=1e-2, vmax=4, zorder=2, s=80, norm=matplotlib.colors.LogNorm())
    #plot black border around the river scatter plot, for readability
    im2 = ax.scatter(fieldMesh_x[i],fieldMesh_y[i], c = r_post_cells_n_avg[i], cmap=cmap_r2, vmin=1e-2, vmax=1.1e-2, zorder=1, s=100)

box = ax.get_position()
ax.set_position([1.35*box.x0, 2.75 * box.y0, box.width * 0.8, box.height * 0.8])

cax1 = fig.add_axes([0.30, 0.22, 0.4, 0.02])
cax2 = fig.add_axes([0.30, 0.14, 0.4, 0.02])
cax3 = fig.add_axes([0.30, 0.06, 0.4, 0.02])


cbar3 = plt.colorbar(plt1,cax=cax3,orientation='horizontal', ticks=[0.001,0.01,0.1,1])
cbar2 = plt.colorbar(plt2,cax=cax2,orientation='horizontal', ticks=[0.001,0.01,0.1,1])
cbar1 = plt.colorbar(plt3,cax=cax1,orientation='horizontal',ticks=[0.01, 0.1, 1, 10])
ax.scatter(3.4, 51.6, marker='X', c='y', s=80, zorder=3) 
cax3.set_title(r'Coastal probabilities [%]')
cax2.set_title(r'Fishery probabilities [%]')   
cax1.set_title(r'River probabilities [%]') 

stop = timeit.default_timer()
print('Time in cell plotting Fig. 2: ', stop - start)
#%% Plotting fig 3
datatotal_temp = np.empty((15,52))

datatotal_temp[0:10,:] = c_post_list_n_temp.copy()
datatotal_temp[10:,:]= f_post_list_n_temp.copy()

labels = ["UK E", "UK SW", "UK SE", "SC", "IR", "NL", "BE", "FR N",  "FR Brit.", "Other (coastal)", "Channel W", "Channel E", "NL", "North Sea", "Other (fishery)"]
colorlist= ['#1f77b4','#d62728','#2ca02c','#8c564b','#bcbd22','#ff7f0e','#9467bd','#7f7f7f','#e377c2','k','#1f77b4','#2ca02c','#ff7f0e','#d62728','w']  
            
fig, ax = plt.subplots(figsize=(10,4))
ax.set_ylim(0,102)
X = np.arange(datatotal_temp.shape[1])
for i in range(10):
    ax.bar(X, datatotal_temp[i],
    bottom = np.sum(datatotal_temp[:i], axis = 0), label=labels[i], color=colorlist[i])
for i in range(10,15):
    if i == 14:
        edgecolor = 'k'
    else:
        edgecolor=None
    ax.bar(X, datatotal_temp[i],
    bottom = np.sum(datatotal_temp[:i], axis = 0), label=labels[i], edgecolor = edgecolor, color=colorlist[i], hatch='///')
           
ax.set_xlabel("Beaching date")
ax.set_ylabel("Source probability [%]")
ax.set_xticks([0, 4, 8, 12, 16, 21, 25, 29, 34, 38, 43, 47])
ax.set_xticklabels(['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1,0.08), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
box3 = ax.get_position()
ax.set_position([box3.x0, box3.y0, box3.width * 0.9, box3.height])
#%% Plotting fig 4
datatotal_age = np.empty((15,24))

datatotal_age[0:10,:] = c_post_list_n_age.copy()
datatotal_age[10:,:]= f_post_list_n_age.copy()
          
fig, ax = plt.subplots(figsize=(10,4))
ax.set_ylim(0,102)
X = np.arange(datatotal_age.shape[1])
for i in range(10):
    ax.bar(X, datatotal_age[i],
    bottom = np.sum(datatotal_age[:i], axis = 0), label=labels[i], color=colorlist[i])
for i in range(10,15):
    if i == 14:
        edgecolor = 'k'
    else:
        edgecolor=None
    ax.bar(X, datatotal_age[i],
    bottom = np.sum(datatotal_age[:i], axis = 0), label=labels[i], edgecolor = edgecolor, color=colorlist[i], hatch='///')
           
ax.set_xlabel("Assumed particle age [months]")
ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
ax.set_ylabel("Source probability [%]")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1,0.08), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
box3 = ax.get_position()
ax.set_position([box3.x0, box3.y0, box3.width * 0.9, box3.height])
