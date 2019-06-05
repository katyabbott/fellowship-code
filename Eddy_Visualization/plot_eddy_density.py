#Density maps of eddies from Bermuda and NWAT data, made for ASLO conference

from netCDF4 import Dataset
import pandas as pd
import numpy as np
import datetime
import time
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from numpy import ma
import matplotlib.colors as colors
import matplotlib

matplotlib.rcParams.update({'font.size': 16})


## User inputs
nc_file = '../cosmo-output/eddy_tracks/rad050_lessthan500km/{}.nc' #User enters in file needed
bermuda_nc = '../../tracks/{}.nc'
image_path = '../../../Helen_Fellowship/Posters_Presentations/ASLO_Poster/images/'
#nc_file = '../input/original_input/nwat_ssh.0010.nc'
cyc = True  #False if anticyclonic, True if cyclonic

## Constants
day_ref = 1721424 #in days, corresponds to year 1, month 1, day 1 (01-01-01)
date_ref = datetime.datetime(year=1, month=1, day=1)

nc_grid = '../nwat_grd_latlon.nc'
ds_grid = Dataset(nc_grid)
#ds = Dataset(nc_file)
lat = ds_grid.variables['lat_rho'][:]
lon = ds_grid.variables['lon_rho'][:]

minlon, maxlon = -80, lon.max()#lon.min(), lon.max()
minlat, maxlat = lat.min(), lat.max()
minlat_sm, minlon_sm = 38, -70
bminlon, bmaxlon = 290.125 - 360, 300.125 - 360
bminlat, bmaxlat = 27.125, 37.125
cmap_fn = lambda x: 'Blues' if x == 'Cyclonic' else 'Reds'

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def jdate_conversion(jdate):
    delta_t = datetime.timedelta(days = jdate - day_ref)
    date = date_ref + delta_t
    date_str = datetime.datetime.strftime(date, '%y-%m-%d')
    return date  #convert from a Julian date to datetime object

def df_from_nc(nc_file, date_fn = None):
    ds = Dataset(nc_file)
    lat = list(ds['lat'][:])  #latitude in degrees
    lon = list(ds['lon'][:])  #longitude in degrees
    radius = list(ds['radius_e'][:]) #radius in km
    amp = list(ds['A'][:])  #amplitude in m (not cm)
    date = list(ds['j1'][:]) #Julian Date, with first date (1721687) corresponding to 01-09-21 (year 1 of model, Sept. 21)
    eddy = list(ds['track'][:])  #eddy ID
    #ext_contour_height = list(ds['height_external_contour'][:]) #height at outer contour
    #inn_contour_height = list(ds['height_inner_contour'][:]) #height at innermost contour of eddy
    obs_number = list(ds['n'][:])
    df = pd.DataFrame([eddy, date, lat, lon, radius, amp,obs_number]).T
    df.columns = ['eddy_ID', 'date', 'lat', 'lon', 'radius', 'amp', 'obs_number']
    df['date'] = df['date'].astype(np.int32)
    if date_fn:
        df['date'] = df['date'].apply(date_fn)
    df['eddy_ID'] = df['eddy_ID'].astype(int)
    ds.close()
    return df  #convert netcdf into pandas dataframe

def contour_plot_data(df, years = 18):
    df['latround'] = df['lat'].apply(lambda x: round(x*2)/2)
    #This bins points .25 degrees or less from the lat point. (bin coordinate is at center of the bin)
    #Another option is using math.floor(x*2)/2 so that bin coordinate is at the lower left corner of bin
    df['lonround'] = df['lon'].apply(lambda x: round(x*2)/2)
    df['latlon'] = list(zip(df['lonround'].astype('float'), 
        df['latround'].astype('float')))
    c_contour = df.groupby(['eddy_ID', 'latlon']).agg({'date': 'count',
        'radius': np.mean, 'amp': np.mean, 'lat': np.mean, 'lon': np.mean, 'latround': 'first', 'lonround': 'first'})
    c_contour.reset_index(inplace = True)
    contour_agg = c_contour.groupby('latlon').agg({'date': 'sum', 'latround': 'first', 'lonround': 'first'})
    lat = list(contour_agg['latround'][:])
    lon = list(contour_agg['lonround'][:])
    freq = list(contour_agg['date'][:]/years)
    return lon, lat, freq

def make_contour_map(x, y, lat_bound, lon_bound):  #lat, lon are tuples
    m = Basemap(projection='merc', llcrnrlat= lat_bound[0],
                urcrnrlat=lat_bound[1], llcrnrlon= lon_bound[0],
                urcrnrlon=lon_bound[1], lat_ts=20, resolution='l')
    x1_m, y1_m = m(x, y)
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)
    # Add States and Country Boundaries
    m.drawstates()
    m.drawcountries()
    m.fillcontinents()
    m.drawmapboundary(fill_color='white')
    return x1_m, y1_m


"""
Plotting eddy frequency at submesoscale
"""

#Process data
df_hires = df_from_nc(nc_file.format('Anticyclonic'), date_fn = jdate_conversion)
df_lores = df_from_nc(bermuda_nc.format('Anticyclonic'))
minrad = df_lores['radius'].min()  #want to only look at eddies with radius smaller than resolved by satellite
ac_hires = df_hires.copy()
ac_hires = ac_hires[ac_hires['radius'] < minrad]
df_hires = df_from_nc(nc_file.format('Cyclonic'), date_fn = jdate_conversion)
cc_hires = df_hires.copy()
cc_hires = cc_hires[cc_hires['radius'] < minrad]
sm_alon, sm_alat, sm_afreq = contour_plot_data(ac_hires) #sm = submesoscale
sm_clon, sm_clat, sm_cfreq = contour_plot_data(cc_hires) #sm = submesoscale
x1, y1 = np.mgrid[minlon_sm:maxlon:1000j, minlat_sm:maxlat:1000j]
x1, y1 = x1.astype('float32'), y1.astype('float32')
smz_c = griddata((sm_clon,sm_clat),sm_cfreq,(x1,y1),method='linear')
smz_a = griddata((sm_alon,sm_alat),sm_afreq,(x1,y1),method='linear')

#Plot 
fig, ax = plt.subplots(figsize = (16,10))
plt.tight_layout()
plt.title('Frequency of submesoscale eddies from NWAT model')
x, y = make_contour_map(x1, y1, (39, maxlat), (minlon_sm, -43.5))
im1 = ax.contourf(x, y, smz_a, cmap = 'Reds')
cb2ax = fig.add_axes([1, 0.09, 0.035, 0.4])  #left, bottom, width, height
fig.colorbar(im1, cax=cb2ax, label='Anticyclonic eddy concentration (counts/year)')
im2 = ax.contourf(x, y, smz_c, alpha = 0.5, cmap = 'Blues')
#plt.colorbar(label='Cyclonic eddy radius (km)', orientation='vertical')
cb1ax = fig.add_axes([1, 0.53, 0.035, 0.4])
fig.colorbar(im2, cax=cb1ax, label='Cyclonic eddy concentration (counts/year)')
plt.savefig(image_path + "freq_eddies_NWAT_submeso.png", bbox_inches='tight')

"""
Plotting density map of anticyclonic and cyclonic radii
"""

#Process data
df = df_from_nc(nc_file.format('Cyclonic'), date_fn = jdate_conversion)
df['latround'] = df['lat'].apply(lambda x: round(x*2)/2)
#This bins points .25 degrees or less from the lat point. (bin coordinate is at center of the bin)
#Another option is using math.floor(x*2)/2 so that bin coordinate is at the lower left corner of bin
df['lonround'] = df['lon'].apply(lambda x: round(x*2)/2)
df['latlon'] = list(zip(df['lonround'].astype('float'), 
    df['latround'].astype('float')))
c_contour = df.groupby(['eddy_ID', 'latlon']).agg({'date': 'count',
    'radius': np.mean, 'amp': np.mean, 'lat': np.mean, 'lon': np.mean, 'latround': 'first', 'lonround': 'first'})
c_contour.reset_index(inplace = True)
contour_agg = c_contour.groupby('latlon').agg({'radius': np.mean, 'latround': 'first', 'lonround': 'first'})
lat = list(contour_agg['latround'][:])
lon = list(contour_agg['lonround'][:])
freq = list(contour_agg['radius'][:])
rad_z_c = griddata((lon, lat), freq, (x1, y1))
rad_z_a = griddata((lon, lat), freq, (x1, y1))

# Plot
fig, ax = plt.subplots(figsize = (15,10))
plt.tight_layout()
plt.title('Radius of eddies in NWAT region')
x, y = make_contour_map(x1, y1, (minlat, maxlat), (minlon, maxlon))
im1 = ax.contourf(x, y, rad_z_a, cmap = 'Reds')
cb2ax = fig.add_axes([.86, 0.04, 0.035, 0.43])  #left, bottom, width, height
fig.colorbar(im1, cax=cb2ax, label='Anticyclonic eddy radius (km)')
#plt.colorbar(label='Anticyclonic eddy radius (km)', pad = -.001, orientation='vertical')
im2 = ax.contourf(x, y, rad_z_c, alpha = .5, cmap = 'Blues')
#plt.colorbar(label='Cyclonic eddy radius (km)', orientation='vertical')
cb1ax = fig.add_axes([0.86, 0.51, 0.035, 0.43])
fig.colorbar(im2, cax=cb1ax, label='Cyclonic eddy radius (km)')
plt.savefig(image_path + 'radius_alleddies_NWAT.png')


"""
Plotting eddy frequency across entire NWAT basin
"""
maxlat = 40 #testing a restricted domain
minlon = -75
x1, y1 = np.mgrid[minlon:maxlon:1000j, minlat:maxlat:1000j]
x1, y1 = x1.astype('float32'), y1.astype('float32')

#Process data
df = df_from_nc(nc_file.format('Cyclonic'), date_fn = jdate_conversion)
clon, clat, cfreq = contour_plot_data(df)
df = df_from_nc(nc_file.format('Anticyclonic'), date_fn = jdate_conversion)
alon, alat, afreq = contour_plot_data(df)
z_c = griddata((clon,clat),cfreq,(x1,y1),method='linear')
z_a = griddata((alon,alat),afreq,(x1,y1),method='linear')

##Plot
plt.figure(figsize=(16,10))
m = Basemap(projection='merc', llcrnrlat= minlat - 1,
            urcrnrlat=maxlat + 1, llcrnrlon= -75 - 1,
            urcrnrlon=maxlon + 1, lat_ts=20, resolution='l')
x1_m, y1_m = m(x1, y1)
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)
# Add States and Country Boundaries
m.drawstates()
m.drawcountries()
m.fillcontinents()
m.drawmapboundary(fill_color='white')
#plt.contourf(x1_m, y1_m,z_c, cmap = 'Blues')
#m.plot(lon[0], lat[0], latlon = True)
plt.contourf(x1_m, y1_m, z_c, cmap = 'Blues')
plt.colorbar(label='Eddy radius (km)', orientation='vertical')
plt.title('Radius of cyclonic eddies in NWAT region')
plt.savefig(image_path + 'images/eddy_rad_NWAT_cyclonic.png')


"""
Plotting eddy frequency in Bermuda (from both NWAT and satellite data)
"""
##Processing data
x1, y1 = np.mgrid[bminlon:bmaxlon:1000j, bminlat:bmaxlat:1000j]
x1, y1 = x1.astype('float32'), y1.astype('float32')
df = df_from_nc(bermuda_nc.format('Cyclonic'), date_fn = jdate_conversion)
bclon, bclat, bcfreq = contour_plot_data(df, years = 25)
df = df_from_nc(bermuda_nc.format('Anticyclonic'), date_fn = jdate_conversion)
balon, balat, bafreq = contour_plot_data(df, years = 25)
bz_a = griddata((bclon,bclat),bcfreq,(x1,y1),method='linear')
bz_c = griddata((balon,balat),bafreq,(x1,y1),method='linear')
nz_c = griddata((clon,clat),cfreq,(x1,y1),method='linear')
nz_a = griddata((alon,alat),afreq,(x1,y1),method='linear')

"""
Masking data to make map look cleaner. Find a way to automate this
l = bz_a.copy()
l[np.isnan(l)] = 1
l[l != 1] = 0

x1m = ma.masked_array(x1, mask = l)
y1m = ma.masked_array(y1, mask = l)

y1m.min()
y1m.max()
x1m.min()
x1m.max()
"""


##Plot data
plt.figure(figsize=(16,10))
plt.tight_layout()
m = Basemap(projection='merc',llcrnrlat=y1m.min(),urcrnrlat=y1m.max(),\
            llcrnrlon=x1m.min(),urcrnrlon=x1m.max(),lat_ts=20,resolution='l', area_thresh=0)
#m = Basemap(projection='merc',llcrnrlat=bminlat,urcrnrlat=bmaxlat,\
            #llcrnrlon=bminlon,urcrnrlon=bmaxlon,lat_ts=20,resolution='l', area_thresh=0)
m.drawparallels(np.arange(-80., 81., 3.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(-180., 181., 3.), labels=[0, 0, 0, 1], fontsize=15)
# Add Bermuda outline
m.drawcoastlines()
m.fillcontinents()
m.drawlsmask(land_color='grey',ocean_color='white',lakes=True)
x1_mask, y1_mask = m(x1m, y1m)
x11, y11 = m(x1, y1)
#plt.contourf(x1_mask, y1_mask,bz_c, cmap = 'Blues')
#plt.contourf(x1_mask, y1_mask,bz_a, cmap = 'Reds')
#plt.contourf(x11, y11,nz_c, cmap = 'Blues')
plt.contourf(x11, y11,nz_a, cmap = 'Reds')
plt.colorbar(label='Eddy concentration (mean counts/year)', orientation='vertical')
plt.title('Frequency of eddies in the Bermuda region \n from ROMS model')
plt.savefig(image_path + 'eddy_freq_Bermuda_NWAT_anticyclonic.png', bbox_inches = 'tight')

#Plot anomaly (sigma)
sigma = (bz_a/5 + bz_c/5) - nz_a - nz_c
a_sigma = bz_a/5 - nz_a
c_sigma = bz_c/5 - nz_c
plt.figure(figsize=(16,10))
plt.tight_layout()
m = Basemap(projection='merc',llcrnrlat=y1m.min(),urcrnrlat=y1m.max(),\
            llcrnrlon=x1m.min(),urcrnrlon=x1m.max(),lat_ts=20,resolution='l', area_thresh=0)
#m = Basemap(projection='merc',llcrnrlat=bminlat,urcrnrlat=bmaxlat,\
            #llcrnrlon=bminlon,urcrnrlon=bmaxlon,lat_ts=20,resolution='l', area_thresh=0)
m.drawparallels(np.arange(-80., 81., 3.), labels=[1, 0, 0, 0], fontsize=15)
m.drawmeridians(np.arange(-180., 181., 3.), labels=[0, 0, 0, 1], fontsize=15)
# Add Bermuda outline
m.drawcoastlines()
m.fillcontinents()
m.drawlsmask(land_color='grey',ocean_color='white',lakes=True)
x1_m, y1_m = m(x1m, y1m)
plt.contourf(x1_m,y1_m,a_sigma, norm=MidpointNormalize(midpoint=0), cmap = 'Spectral')
plt.colorbar(label = 'Normalized difference in eddy concentration', extend='min')
plt.title('Difference between anticyclonic eddy concentrations \n in NWAT and satellite data')
plt.savefig(image_path + 'sigma_anticyc.png', bbox_inches = 'tight')



