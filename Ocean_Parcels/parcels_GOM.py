from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, random, \
    ErrorCode, ParticleFile, Variable, plotTrajectoriesFile
from datetime import timedelta as delta
import datetime
import numpy as np
import math
import pandas as pd
from netCDF4 import Dataset
ds = Dataset(ncfile)
ds.variables.keys()

ds.variables['w_velocity']

ncfile = 'GOM_HYCOM_expt31.0_daily_uvwST_04012014_05312014.nc'
ncfile_startDate = datetime.datetime(2014, 4, 1) #first entry in nc file
locations = pd.read_csv('all_release_sites.csv', parse_dates = ['Date'])
startLon, startLat = locations.loc[1]['Lon'], locations.loc[1]['Lat']
startDate = locations.loc[1]['Date'].to_pydatetime()
nParticles = 100
nSites = len(locations)

def time_conversion(date, startdate):
    days_since = (date - startdate).days
    return days_since*(60**2)*24 #in seconds

def SampleTemp(particle, fieldset, time):
    particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]

def SampleS(particle, fieldset, time):
    particle.s = fieldset.S[time, particle.depth, particle.lat, particle.lon]

def SampleW(particle, fieldset, time):
    particle.w = fieldset.W[time, particle.depth, particle.lat, particle.lon]

def BrownianDiffusion(particle, fieldset, time):
    kh_zonal = fieldset.Kh / math.pow(1000. * 1.852 * 60. * math.cos(particle.lat * M_PI / 180), 2)
    kh_meridional = fieldset.Kh / math.pow(1000.0 * 1.852 * 60.0, 2)
    # 1000*1.852*60 is conversion from meters to degrees, with cos(lat) adjusting for 
    ## change in longitude arc length with latitude
    r = 1/3. #variance with a sample space [-1,1]
    # Assuming Kh doesn't vary in time or space
    particle.lat += random.uniform(-1., 1.)*math.sqrt(2*math.fabs(particle.dt)*kh_meridional/r)
    particle.lon += random.uniform(-1., 1.)*math.sqrt(2*math.fabs(particle.dt)*kh_zonal/r)

def DeleteParticle(particle, fieldset, time):
    particle.delete()
    
filenames = {'U': ncfile, 'V': ncfile, 'temp': ncfile, 'S': ncfile, 'W': ncfile}
variables = {'U': 'u', 'V': 'v', 'temp': 'temperature', 'S': 'salinity', 'W': 'w_velocity'}
dimensions = {'lat': 'Latitude', 'lon': 'Longitude', 'time': 'MT', 'depth': 'Depth'}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, deferred_load = False)
fieldset.temp.data[fieldset.temp.data == 0] = np.nan
fieldset.Kh = 100.

## How to view different fieldset dimensions
#fieldset.U.grid.time #(or time_full if deferred_load = True)
#fieldset.U.lat (or lon or depth)

class BluefinParticle(JITParticle):
    temp = Variable('temp', initial = fieldset.temp)
    s = Variable('s', initial = fieldset.S)
    w = Variable('w', initial = fieldset.W)

def backtrack_bluefin(start_date, start_lon, start_lat, output):
    pset = ParticleSet.from_list(fieldset = fieldset, pclass = BluefinParticle, 
                        lon = np.repeat(start_lon, nParticles), lat = np.repeat(start_lat,nParticles), 
                        time = time_conversion(start_date, ncfile_startDate))

    kernels = AdvectionRK4 + pset.Kernel(BrownianDiffusion) + pset.Kernel(SampleTemp) + \
        pset.Kernel(SampleS) + pset.Kernel(SampleW)

    pset.execute(kernels, runtime=delta(days=10), dt = -delta(minutes=5), 
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
        output_file=pset.ParticleFile(name = output, outputdt=delta(hours=1)))

for index, row in locations.iterrows():
    start_date = row['Date'].to_pydatetime()
    start_lon, start_lat = row['Lon'], row['Lat']
    location = row['Location']
    output = 'trajectories/{1}_Bluefin_trajectories_{0}_2.nc'.format(index, location)
    backtrack_bluefin(start_date, start_lon, start_lat, output)

