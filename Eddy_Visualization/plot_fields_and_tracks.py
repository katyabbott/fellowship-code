#Use to plot ocean fields (figure 1) and eddy tracks (figure 2) 

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from numpy import ma
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cmocean
from collections import defaultdict
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from mpl_toolkits.axes_grid1 import make_axes_locatable


image_path = '../../Posters_Presentations/ASLO_Poster/images/'
bminlon, bmaxlon = 290.125 - 360, 300.125 - 360
bminlat, bmaxlat = 27.125, 37.125

nc_grid = 'model/nwat_grd_latlon.nc'
ds_grid = Dataset(nc_grid)
nc_file_ssh = 'model/input/original_input/nwat_ssh.0900.nc' #arbitrary date
ssh_data = Dataset(nc_file_ssh)
ssh = ssh_data.variables['zeta'][0]
mask = ssh_data.variables['mask_rho'][:]
lat = ssh_data.variables['lat_rho'][:]
lon = ssh_data.variables['lon_rho'][:]
minlon, maxlon = lon.min(), lon.max()
minlat, maxlat = lat.min(), lat.max()
ssh_masked = ma.masked_array(ssh, mask = ~mask.astype('bool'))

#Figure 1 (SSH)
fig, ax = plt.subplots(figsize=(19,13))
#Change resolution to 'f' or 'h' for high resolution image (much slower)
m = Basemap(projection='merc', llcrnrlat= minlat - .5,
            urcrnrlat=maxlat + .5, llcrnrlon= -83 - .5,
            urcrnrlon=maxlon + .5, lat_ts=20, resolution='c', area_thresh=0)

#Transform x and y coordinates of SLA to map coordinates
x, y = m(lon, lat)
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=15)
# Add States and Country Boundaries
m.fillcontinents(lake_color = 'azure')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()
## Plot SLA
plt.pcolormesh(x,y, ssh_masked, cmap = cmocean.cm.balance)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="5%", pad=0.5)
plt.colorbar(label = 'Sea level anomaly in meters', cax = cax1)
#plt.savefig(image_path + 'map-of-region_AoT.tif', bbox_inches = 'tight', dpi = 300)


#Figure 2 (eddy tracks)
filepath = "model/cosmo-output/eddy_tracks/rad050_lessthan500km/{}.nc"
names = ['Anticyclonic', 'Cyclonic']
num_obs = 5 #only plots eddy tracks longer than this. 
#Note that for NWAT, 5 observations is 25 days

#Import and reshape data for plotting
def dfnc(nc_file):
    '''
    This takes an nc dataset and creates a Pandas dataframe
    with columns eddy, obs number of eddy and position
    '''
    nc = Dataset(nc_file)
    latlon = list(zip(nc['lat'][:].data,nc['lon'][:].data))
    eddy = list(nc['track'][:].data)
    obs_number = list(nc['n'][:].data)
    df = pd.DataFrame([eddy,obs_number,latlon]).T
    df.columns=['eddy', 'obs_number', 'position']
    return df

for type in names:
    df = dfnc(filepath.format(type))
    #df = truecol(df, bounds)
    eddy = defaultdict(list)
    df.apply(lambda row: eddy[row['eddy']].append(row['position']), axis=1)
    if type == 'Cyclonic':
        c_eddy = eddy
    else:
        a_eddy = eddy

#Plot data
fig = plt.figure(figsize=(19,13))
#Make map
m = Basemap(projection='merc', llcrnrlat= minlat - .5,
            urcrnrlat=maxlat + .5, llcrnrlon= -83 - .5,
            urcrnrlon=maxlon + .5, lat_ts=20, resolution='c', area_thresh=0)
m.drawparallels(np.arange(-80., 81., 10), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10), labels=[0, 0, 0, 1], fontsize=10)
# Add States and Country Boundaries
m.drawstates()
m.drawcountries()
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary(fill_color='white')

#Plot eddy tracks 
for k, v in c_eddy.items():
    if len(v) > num_obs:  
        lons = [i[1] for i in v]
        lats = [i[0] for i in v]
        x, y = m(lons, lats)
        cc = m.plot(x, y, linewidth=.5, color='blue', alpha=.8, label = 'Cyclonic')
for k, v in a_eddy.items():
    if len(v) > num_obs:
        lons = [i[1] for i in v]
        lats = [i[0] for i in v]
        x, y = m(lons, lats)       
        ac = m.plot(x, y, linewidth=.5, color='red', alpha=.8, label = 'Anticyclonic')
        
        