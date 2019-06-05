import pandas as pd
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import glob
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
plt.style.use('seaborn-darkgrid')

#matplotlib.rcdefaults()
east = False

west_list = glob.glob('trajectories/West*trajectories_*_2.nc')
east_list = glob.glob('trajectories/East*_2.nc')
locations = pd.read_csv('all_release_sites.csv', parse_dates = ['Date'])
west_release_lat = list(locations['Lat'][locations['Location'] == 'West-GOM'])
west_release_lon = list(locations['Lon'][locations['Location'] == 'West-GOM'])
east_release_lat = list(locations['Lat'][locations['Location'] == 'East-GOM'])
east_release_lon = list(locations['Lon'][locations['Location'] == 'East-GOM'])

if east:
    release_lat = east_release_lat
    release_lon = east_release_lon
    filelist = east_list
else:
    release_lat = west_release_lat
    release_lon = west_release_lon
    filelist = west_list
#Data extraction
lon = np.empty((0, 241))
lat = lon.copy()
temp = lon.copy()
w = lon.copy()
s = lon.copy()
for ncfile in filelist:
    ds = Dataset(ncfile)
    lon_ds = ds.variables['lon']
    lat_ds = ds.variables['lat']
    temp_ds = ds.variables['temp']
    s_ds = ds.variables['s'] #salinity
    w_ds = ds.variables['w'] #vertical velocity
    lon = np.append(lon, lon_ds, axis = 0)
    lat = np.append(lat, lat_ds, axis = 0)
    temp = np.append(temp, temp_ds, axis = 0)
    s = np.append(s, s_ds, axis = 0)
    w = np.append(w, w_ds, axis = 0)

time = np.arange(0, 24*10+1, 1)  #time elapsed in hours
time_days = time/24

meanT_east = np.nanmean(temp, axis = 0)
stdT_east = np.nanstd(temp, axis = 0)
meanS_east = np.nanmean(s, axis = 0)
stdS_east = np.nanstd(s, axis = 0)
meanW_east = np.nanmean(w, axis = 0)
stdW_east = np.nanstd(w, axis = 0)

meanT_west = np.nanmean(temp, axis = 0)
stdT_west = np.nanstd(temp, axis = 0)
meanS_west = np.nanmean(s, axis = 0)
stdS_west = np.nanstd(s, axis = 0)
meanW_west = np.nanmean(w, axis = 0)
stdW_west = np.nanstd(w, axis = 0)

"""
#salinity colors:
##e3c824
#2d1b86
#temp colors:
#sns.xkcd_rgb['dark salmon']
sns.xkcd_rgb['azure']
"""


plt.figure(figsize = (10, 7))
plt.plot(time_days, meanW_east*1000, '-', color=sns.xkcd_rgb['azure'], label = 'East GOM')
plt.fill_between(time_days, meanW_east*1000 + stdW_east*1000, meanW_east*1000 - stdW_east*1000, 
    color=sns.xkcd_rgb['azure'], alpha=0.2)
plt.plot(time_days, meanW_west*1000, '-', color=sns.xkcd_rgb['dark salmon'], label = 'West GOM')
plt.fill_between(time_days, meanW_west*1000 + stdW_west*1000, meanW_west*1000 - stdW_west*1000, 
    color=sns.xkcd_rgb['dark salmon'], alpha=0.2)
plt.xlim(10,0)
#plt.ylim(22.1, 27)
plt.xlabel('Days before sampling')
plt.ylabel('Average upward velocity (mm/s)')
plt.legend(loc='upper left')
plt.savefig('W_velocity_profile_bluefin.png', dpi = 300)


#Visualize trajectories
tmin = np.nanmin(temp)
tmax = np.nanmax(temp)

plt.figure(figsize = (15,15))
tnorm = plt.Normalize(vmin=tmin, vmax=tmax)
sm = plt.cm.ScalarMappable(norm=tnorm)
m = Basemap(projection='merc', 
                 llcrnrlon=-99,
                 llcrnrlat=16,
                 urcrnrlon=-79,
                 urcrnrlat=31,
                 resolution='c')
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=17)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=17)
# Add States and Country Boundaries
#m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary(fill_color='white')
m.drawcoastlines()
m.fillcontinents(color='bisque',lake_color='aqua')
# plot each trajectory separately, in gold color
# documentation: https://matplotlib.org/basemap/index.html
m.scatter(release_lon, release_lat, s = 50, marker = 'X', latlon = True, color = 'red', zorder = 10)
for i in range(len(lon)):
    lons = lon[:][i]
    lats = lat[:][i]
    temps = temp[:][i]
    traj = m.scatter(np.array(lons), np.array(lats), c = temps,  norm = tnorm, latlon=True)
cb = plt.colorbar(traj, fraction=0.038, pad=0.04)
cb.set_label('degrees C')
plt.savefig('West-GOM_trajectories.png')    
