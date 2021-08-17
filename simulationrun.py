from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, Field, VectorField, Variable
import numpy as np
from datetime import timedelta, datetime
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar 
import math
import parcels.rng as ParcelsRandom

file_coast = 'Datafiles_Mikael//datafile_coastMask_297x_375y'
coastMask  = np.genfromtxt(file_coast, delimiter=None)

# 1 is Brittany, 2 is Aberdeen, 3 is Domburg (backwards)
startlocation = 3
startdate = '2020-01-01'
runtime_days = 730
#Enter 0 to include Stokes drift, enter 1 to exclude
stokes = 0
#Enter 0 to include tides, enter 1 to exclude
tides = 0
#value for K based on Neumann, 13.39
K = 13.39

current_data = xr.open_mfdataset('/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/2017*.nc')
lons = current_data.coords['longitude'].values
lats = current_data.coords['latitude'].values
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)

day_start = datetime(2020,1,1,12,00)

startplace = 'Domburg'
startlon = 3.493
startlat = 51.566
#fw = -1 means a backward simulation
fw = -1

outfile = str(startplace + '_'+ startdate + '_Stokes='+ str(stokes)+'_' + str(runtime_days))

#find nearest coastal cell to defined beaching location, to release in water        
def nearestcoastcell(lon,lat):
    dist = np.sqrt((fieldMesh_x - lon)**2 * coastMask + (fieldMesh_y - lat)**2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))
    startlon_release = fieldMesh_x[coords]
    endlon_release = fieldMesh_x[coords[0], coords[1] + 1]
    startlat_release = fieldMesh_y[coords]
    endlat_release = fieldMesh_y[coords[0] + 1, coords[1]]
    return startlon_release, endlon_release, startlat_release, endlat_release, coords

startlon_release, endlon_release, startlat_release, endlat_release, coords = nearestcoastcell(startlon,startlat)
#10x10 particles -> 100 particles homogeneously spread over grid cell
re_lons = np.linspace(startlon_release, endlon_release, 10)
re_lats = np.linspace(startlat_release, endlat_release, 10)
fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)
#%%
variables_surface = {'U': 'uo',
             'V': 'vo'}
dimensions_surface = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}

variables_stokes = {'U_Stokes': 'VSDX',
             'V_Stokes': 'VSDY'}
dimensions_stokes = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}

if stokes == 0:    
    fieldset = FieldSet.from_netcdf("/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/201*.nc", variables_surface, dimensions_surface, allow_time_extrapolation=True)

    fieldset_Stokes = FieldSet.from_netcdf("/data/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_WAV_001_032_NWSHELF_Stokes/201*.nc", variables_stokes, dimensions_stokes, allow_time_extrapolation=True)
    fieldset_Stokes.U_Stokes.units = GeographicPolar()
    fieldset_Stokes.V_Stokes.units = Geographic()
    
    fieldset.add_field(fieldset_Stokes.U_Stokes)
    fieldset.add_field(fieldset_Stokes.V_Stokes)
    
    vectorField_Stokes = VectorField('UV_Stokes',fieldset.U_Stokes,fieldset.V_Stokes)
    fieldset.add_vector_field(vectorField_Stokes)
elif stokes == 1:
    fieldset = FieldSet.from_netcdf("/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/201*.nc", variables_surface, dimensions_surface, allow_time_extrapolation=True)

class PlasticParticle(JITParticle):
    age = Variable('age', dtype=np.float32, initial=0., to_write=True)
    # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach, 5 out of bounds
    beached = Variable('beached',dtype=np.int32,initial=0., to_write=False)
    U_tide = Variable('U_tide', dtype=np.float32, initial=0., to_write=False)    
    V_tide = Variable('V_tide', dtype=np.float32, initial=0., to_write=False)   
#%%
#---------------unbeaching
file_landCurrent_U = 'Datafiles_Mikael/datafile_landCurrentU_%ix_%iy' % (len(lons),len(lats))
file_landCurrent_V = 'Datafiles_Mikael/datafile_landCurrentV_%ix_%iy' % (len(lons),len(lats))

landCurrent_U = np.loadtxt(file_landCurrent_U)
landCurrent_V = np.loadtxt(file_landCurrent_V)

U_land = Field('U_land',landCurrent_U,lon=lons,lat=lats,fieldtype='U',mesh='spherical')
V_land = Field('V_land',landCurrent_V,lon=lons,lat=lats,fieldtype='V',mesh='spherical')


fieldset.add_field(U_land)
fieldset.add_field(V_land)

vectorField_unbeach = VectorField('UV_unbeach',U_land,V_land)
fieldset.add_vector_field(vectorField_unbeach)
#-----------------misc fields

K_m = K*np.ones(fieldMesh_x.shape)
K_z = K*np.ones(fieldMesh_x.shape)


Kh_meridional = Field('Kh_meridional', K_m,lon=lons,lat=lats,mesh='spherical')
Kh_zonal = Field('Kh_zonal', K_z,lon=lons,lat=lats,mesh='spherical')

fieldset.add_field(Kh_meridional)
fieldset.add_field(Kh_zonal)

def TidalMotionM2S2K1O1(particle, fieldset, time):
    """
    Kernel that calculates tidal currents U and V due to M2, S2, K1 and O1 tide at particle location and time
    and advects the particle in these currents (using Euler forward scheme)
    Calculations based on Doodson (1921) and Schureman (1958)
    Also: Sterl (2019)
    """  
    if particle.beached != 5:      
        # Number of Julian centuries that have passed between t0 and time
        t = ((time + fieldset.t0rel)/86400.0)/36525.0
        
        # Define constants to compute astronomical variables T, h, s, N (all in degrees) (source: FES2014 code)
        cT0 = 180.0
        ch0 = 280.1895
        cs0 = 277.0248
        cN0 = 259.1568; cN1 = -1934.1420
        deg2rad = math.pi/180.0
        
        # Calculation of factors T, h, s at t0 (source: Doodson (1921))
        T0 = math.fmod(cT0, 360.0) * deg2rad
        h0 = math.fmod(ch0, 360.0) * deg2rad
        s0 = math.fmod(cs0, 360.0) * deg2rad
        
        # Calculation of V(t0) (source: Schureman (1958))
        V_M2 = 2*T0 + 2*h0 - 2*s0
        V_S2 = 2*T0
        V_K1 = T0 + h0 - 0.5*math.pi
        V_O1 = T0 + h0 - 2*s0 + 0.5*math.pi
        #these are added for nonlinear (Bram)
        V_M4 = 4*T0 - 4*s0 + 4*h0
        V_MS4 = 4*T0 - 2*s0 + 2*h0
        V_S4 = 4*T0
        
        # Calculation of factors N, I, nu, xi at time (source: Schureman (1958))
        # Since these factors change only very slowly over time, we take them as constant over the time step dt
        N = math.fmod(cN0 + cN1*t, 360.0) * deg2rad
        I = math.acos(0.91370 - 0.03569*math.cos(N))
        tanN = math.tan(0.5*N)
        at1 = math.atan(1.01883 * tanN)
        at2 = math.atan(0.64412 * tanN)
        nu = at1 - at2
        xi = -at1 - at2 + N
        nuprim = math.atan(math.sin(2*I) * math.sin(nu)/(math.sin(2*I)*math.cos(nu) + 0.3347))
        
        # Calculation of u, f at current time (source: Schureman (1958))
        u_M2 = 2*xi - 2*nu
        f_M2 = (math.cos(0.5*I))**4/0.9154
        u_S2 = 0
        f_S2 = 1
        u_K1 = -nuprim
        f_K1 = math.sqrt(0.8965*(math.sin(2*I))**2 + 0.6001*math.sin(2*I)*math.cos(nu) + 0.1006)
        u_O1 = 2*xi - nu
        f_O1 = math.sin(I)*(math.cos(0.5*I))**2/0.3800
        #these are added for nonlinear (Bram)
        u_M4 = 4*xi - 4*nu
        f_M4 = (f_M2)**2
        u_MS4 = 2*xi - 2*nu
        f_MS4 = f_M2
        u_S4 = 0
        f_S4 = 1

        
        # Euler forward method to advect particle in tidal currents
    
        lon0, lat0 = (particle.lon, particle.lat)
    
        # Zonal amplitudes and phaseshifts at particle location and time
        Uampl_M2_1 = f_M2 * fieldset.UaM2[time, particle.depth, lat0, lon0]
        Upha_M2_1 = V_M2 + u_M2 - fieldset.UgM2[time, particle.depth, lat0, lon0]
        Uampl_S2_1 = f_S2 * fieldset.UaS2[time, particle.depth, lat0, lon0]
        Upha_S2_1 = V_S2 + u_S2 - fieldset.UgS2[time, particle.depth, lat0, lon0]
        Uampl_K1_1 = f_K1 * fieldset.UaK1[time, particle.depth, lat0, lon0]
        Upha_K1_1 = V_K1 + u_K1 - fieldset.UgK1[time, particle.depth, lat0, lon0]
        Uampl_O1_1 = f_O1 * fieldset.UaO1[time, particle.depth, lat0, lon0]
        Upha_O1_1 = V_O1 + u_O1 - fieldset.UgO1[time, particle.depth, lat0, lon0]
        
        #nonlinear, Bram
        Uampl_M4_1 = f_M4 * fieldset.UaM4[time, particle.depth, lat0, lon0]
        Upha_M4_1 = V_M4 + u_M4 - fieldset.UgM4[time, particle.depth, lat0, lon0]
        Uampl_MS4_1 = f_MS4 * fieldset.UaMS4[time, particle.depth, lat0, lon0]
        Upha_MS4_1 = V_MS4 + u_MS4 - fieldset.UgMS4[time, particle.depth, lat0, lon0]
        Uampl_S4_1 = f_S4 * fieldset.UaS4[time, particle.depth, lat0, lon0]
        Upha_S4_1 = V_S4 + u_S4 - fieldset.UgS4[time, particle.depth, lat0, lon0]        
        
        # Meridional amplitudes and phaseshifts at particle location and time
        Vampl_M2_1 = f_M2 * fieldset.VaM2[time, particle.depth, lat0, lon0]
        Vpha_M2_1 = V_M2 + u_M2 - fieldset.VgM2[time, particle.depth, lat0, lon0]
        Vampl_S2_1 = f_S2 * fieldset.VaS2[time, particle.depth, lat0, lon0]
        Vpha_S2_1 = V_S2 + u_S2 - fieldset.VgS2[time, particle.depth, lat0, lon0]
        Vampl_K1_1 = f_K1 * fieldset.VaK1[time, particle.depth, lat0, lon0]
        Vpha_K1_1 = V_K1 + u_K1 - fieldset.VgK1[time, particle.depth, lat0, lon0]
        Vampl_O1_1 = f_O1 * fieldset.VaO1[time, particle.depth, lat0, lon0]
        Vpha_O1_1 = V_O1 + u_O1 - fieldset.VgO1[time, particle.depth, lat0, lon0]
        
        #nonlinear, Bram
        Vampl_M4_1 = f_M4 * fieldset.VaM4[time, particle.depth, lat0, lon0]
        Vpha_M4_1 = V_M4 + u_M4 - fieldset.VgM4[time, particle.depth, lat0, lon0]
        Vampl_MS4_1 = f_MS4 * fieldset.VaMS4[time, particle.depth, lat0, lon0]
        Vpha_MS4_1 = V_MS4 + u_MS4 - fieldset.VgMS4[time, particle.depth, lat0, lon0]
        Vampl_S4_1 = f_S4 * fieldset.VaS4[time, particle.depth, lat0, lon0]
        Vpha_S4_1 = V_S4 + u_S4 - fieldset.VgS4[time, particle.depth, lat0, lon0]
        
        # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
        Uvel_M2_1 = Uampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Upha_M2_1)
        Uvel_S2_1 = Uampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Upha_S2_1)
        Uvel_K1_1 = Uampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Upha_K1_1)
        Uvel_O1_1 = Uampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Upha_O1_1)
        
        #nonlinear, Bram
        Uvel_M4_1 = Uampl_M4_1 * math.cos(fieldset.omegaM4 * (time + fieldset.t0rel) + Upha_M4_1)
        Uvel_MS4_1 = Uampl_MS4_1 * math.cos(fieldset.omegaMS4 * (time + fieldset.t0rel) + Upha_MS4_1)
        Uvel_S4_1 = Uampl_S4_1 * math.cos(fieldset.omegaS4 * (time + fieldset.t0rel) + Upha_S4_1)
        
        Vvel_M2_1 = Vampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Vpha_M2_1)
        Vvel_S2_1 = Vampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Vpha_S2_1)
        Vvel_K1_1 = Vampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Vpha_K1_1)
        Vvel_O1_1 = Vampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Vpha_O1_1)
        
        #nonlinear, Bram
        Vvel_M4_1 = Vampl_M4_1 * math.cos(fieldset.omegaM4 * (time + fieldset.t0rel) + Vpha_M4_1)
        Vvel_MS4_1 = Vampl_MS4_1 * math.cos(fieldset.omegaMS4 * (time + fieldset.t0rel) + Vpha_MS4_1)
        Vvel_S4_1 = Vampl_S4_1 * math.cos(fieldset.omegaS4 * (time + fieldset.t0rel) + Vpha_S4_1)
        
        # Total zonal and meridional velocity, only linear
#        U1 = Uvel_M2_1 + Uvel_S2_1 + Uvel_K1_1 + Uvel_O1_1 # total zonal velocity
#        V1 = Vvel_M2_1 + Vvel_S2_1 + Vvel_K1_1 + Vvel_O1_1 # total meridional velocity

        # Total zonal and meridional velocity, including nonlinear        
        U1 = Uvel_M2_1 + Uvel_S2_1 + Uvel_K1_1 + Uvel_O1_1 + Uvel_M4_1 + Uvel_MS4_1 + Uvel_S4_1 # total zonal velocity
        V1 = Vvel_M2_1 + Vvel_S2_1 + Vvel_K1_1 + Vvel_O1_1 + Vvel_M4_1 + Vvel_MS4_1 + Vvel_S4_1 # total meridional velocity
        
        particle.U_tide = U1
        particle.V_tide = V1
        # New lon + lat
        particle.lon += U1*particle.dt
        particle.lat += V1*particle.dt
    else:
        particle.lon += 0
        particle.lat += 0
    
#freeze out of bounds particles at discrete (randomly chosen) location instead of deleting; to prevent problems when concatenating in analysis
#not an elegant solution, but it works
def OutOfBounds(particle, fieldset, time): 
    particle.lon = -17.43
    particle.lat = 62.65
    #if particle.beached = 5, it is not advected anymore by any kernel and is thus frozen
    particle.beached = 5

def AdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
    Function needs to be converted to Kernel object before execution"""
    if particle.beached != 5:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    else:
        particle.lon += 0
        particle.lat += 0

    
def StokesUV(particle, fieldset, time):
    if particle.beached == 0:
            (u_uss, v_uss) = fieldset.UV_Stokes[time, particle.depth, particle.lat, particle.lon]
            particle.lon += u_uss * particle.dt
            particle.lat += v_uss * particle.dt
            particle.beached = 3

def DiffusionUniformKh(particle, fieldset, time):
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.
    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
        fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
        fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)
    where mesh is either 'flat' or 'spherical'
    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a seperate
    advection kernel.
    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    if particle.beached != 5:
        # Wiener increment with zero mean and std of sqrt(dt)
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    
        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])
    
        particle.lon += bx * dWx
        particle.lat += by * dWy
       
        particle.beached = 3 
    else:
        particle.lon += 0
        particle.lat += 0
            
def BeachTesting(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        else:
            particle.beached = 0
            
def UnBeaching(particle, fieldset, time):
    if particle.beached == 4 or particle.beached == 1:
        #-1 for backwards, otherwise the current is landward instead of seaward for unbeaching
        dtt = -1*particle.dt
        (u_land, v_land) = fieldset.UV_unbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_land * dtt
        particle.lat += v_land * dtt
        particle.beached = 0
        
def Ageing(particle, fieldset, time):      
    particle.age += particle.dt   

#-----------------tides--------------
t0 = datetime(1900,1,1,0,0) # origin of time = 1 January 1900, 00:00:00 UTC
fieldset.add_constant('t0rel', (day_start - t0).total_seconds()) # number of seconds elapsed between t0 and starttime   
files_eastward = '/data/oceanparcels/input_data/FES2014Data/eastward_velocity/'
files_northward = '/data/oceanparcels/input_data/FES2014Data/northward_velocity/'
    
deg2rad = math.pi/180.0 # factor to convert degrees to radians

def create_fieldset_tidal_currents(name,filename):
    '''
    Create fieldset for a given type of tide (name = M2, S2, K1, or O1)
    '''
    filename_U = files_eastward + '%s.nc' %filename
    filename_V = files_northward + '%s.nc' %filename
       
    filenames = {'Ua%s'%name: filename_U,
                 'Ug%s'%name: filename_U,
                 'Va%s'%name: filename_V,
                 'Vg%s'%name: filename_V}
    variables = {'Ua%s'%name: 'Ua',
                 'Ug%s'%name: 'Ug',
                 'Va%s'%name: 'Va',
                 'Vg%s'%name: 'Vg'}
    dimensions = {'lat': 'lat',
                  'lon': 'lon'}
    
    fieldset_tmp = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical')
    
    exec('fieldset_tmp.Ua%s.set_scaling_factor(1e-2)'%name)
    exec('fieldset_tmp.Ug%s.set_scaling_factor(deg2rad)'%name) # convert from degrees to radians
    exec('fieldset_tmp.Va%s.set_scaling_factor(1e-2)'%name) #cm/s to m/s
    exec('fieldset_tmp.Vg%s.set_scaling_factor(deg2rad)'%name)
    
    exec('fieldset_tmp.Ua%s.units = GeographicPolar()'%name)
    exec('fieldset_tmp.Va%s.units = Geographic()'%name)
    
    exec('fieldset.add_field(fieldset_tmp.Ua%s)'%name)
    exec('fieldset.add_field(fieldset_tmp.Ug%s)'%name)
    exec('fieldset.add_field(fieldset_tmp.Va%s)'%name)
    exec('fieldset.add_field(fieldset_tmp.Vg%s)'%name)

create_fieldset_tidal_currents('M2','conv_m2')
create_fieldset_tidal_currents('S2','conv_s2')
create_fieldset_tidal_currents('K1','conv_k1')
create_fieldset_tidal_currents('O1','conv_o1')
create_fieldset_tidal_currents('M4','conv_m4')
create_fieldset_tidal_currents('MS4','conv_ms4')
create_fieldset_tidal_currents('S4','conv_s4')

omega_M2 = 28.9841042 # angular frequency of M2 in degrees per hour
fieldset.add_constant('omegaM2', (omega_M2 * deg2rad) / 3600.0) # angular frequency of M2 in radians per second

omega_S2 = 30.0000000 # angular frequency of S2 in degrees per hour
fieldset.add_constant('omegaS2', (omega_S2 * deg2rad) / 3600.0) # angular frequency of S2 in radians per second

omega_K1 = 15.0410686 # angular frequency of K1 in degrees per hour
fieldset.add_constant('omegaK1', (omega_K1 * deg2rad) / 3600.0) # angular frequency of K1 in radians per second

omega_O1 = 13.9430356 # angular frequency of O1 in degrees per hour
fieldset.add_constant('omegaO1', (omega_O1 * deg2rad) / 3600.0) # angular frequency of O1 in radians per second 

#source for summation of frequencies (Andersen, 1999)
omega_M4 = omega_M2 + omega_M2 # angular frequency of S2 in degrees per hour
fieldset.add_constant('omegaM4', (omega_M4 * deg2rad) / 3600.0) # angular frequency of S2 in radians per second

omega_MS4 = omega_M2 + omega_S2 # angular frequency of K1 in degrees per hour
fieldset.add_constant('omegaMS4', (omega_MS4 * deg2rad) / 3600.0) # angular frequency of K1 in radians per second

omega_S4 = omega_S2 + omega_S2 # angular frequency of O1 in degrees per hour
fieldset.add_constant('omegaS4', (omega_S4 * deg2rad) / 3600.0) # angular frequency of O1 in radians per second  
#%%
#release particles for two years and advect for two years.
#releasing three times separately, to prevent having to advect once for 6 years resulting in long runtimes and large files with lots of redundant data
startdatelist = ['2020-01-01', '2018-01-01', '2016-01-01']

for i in range(3):
    startdate_i = startdatelist[i]
    if i == 2:
        runtime_releasedays = int(runtime_days/2)
    else: 
        #the last release in 2016-01-01 is only releasing particles for one year. 
        #Since a particle released on 2015-01-01 is backtracked until 2013-01-01 (end of current data availability) 
        runtime_releasedays = runtime_days
        
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                                 time = np.datetime64(startdate_i),
                                 repeatdt=timedelta(hours=24).total_seconds(),
                                 lon = fieldMesh_x_re,
                                 lat = fieldMesh_y_re)
    
    output_file = pset.ParticleFile(name="results/{}.nc".format(outfile+str(i)), outputdt=timedelta(hours=24))
      

    if tides == 0: 
        #kernels with tides        
        kernels = (pset.Kernel(AdvectionRK4) + pset.Kernel(StokesUV) + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching)
                + pset.Kernel(TidalMotionM2S2K1O1) + pset.Kernel(DiffusionUniformKh)  +  pset.Kernel(Ageing)
                + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching)) 
        
    elif tides == 1:
        #kernels without tides      
        kernels = (pset.Kernel(AdvectionRK4) + pset.Kernel(StokesUV) + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching)
                    + pset.Kernel(DiffusionUniformKh)  +  pset.Kernel(Ageing)
                    + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching))       
    
    pset.execute(kernels,
                 #-1 because you repeat, first one is not taken into account. Prevents too many particles from being released.
                 runtime=timedelta(days=(runtime_releasedays - 1)),
                 dt=fw*timedelta(hours=2),
                 output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})
    #%%
    output_file.close()
    #after particles have been released for two years, simulate these for two more years without advecting
    #to make sure that all particles have been advected for at least two years
    pset = ParticleSet.from_particlefile(fieldset=fieldset, pclass=PlasticParticle,
                                          filename="results/{}.nc".format(outfile+str(i)), restart=True, restarttime = np.nanmin)
    
    output_file2 = pset.ParticleFile(name="results/{}2.nc".format(outfile+str(i)), outputdt=timedelta(hours=24))
    
    pset.execute(kernels,
                 runtime=timedelta(days=runtime_days),
                 dt=fw*timedelta(hours=2),
                 output_file=output_file2,
                 recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})
    
    output_file2.close()