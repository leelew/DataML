import os
import math
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import MultipleLocator
from pylab import rcParams
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.basemap import Basemap

font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 16,
          'legend.frameon': False,
          'xtick.labelsize': 16,
          'xtick.direction': 'out',
          'ytick.labelsize': 16,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

# models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM_2.2', 'GLDAS_Noah_2.1', 'ERA5', 'ET_3T', 'EB_ET', 'PMLV2', 'FLUXCOM_9km',
#           'FLUXCOM']
models = ['FLUXCOM_9km', 'FLUXCOM']
'''
ET_3T/PLMV2 remap to delete sea area (done)
only 20% data of left (done)
num:[r2, r, rmse]
'''

path = "/tera07/zhwei/For_QingChen/DataML/output/forecast/"
pathout = "/tera07/zhwei/For_QingChen/DataML/plot/fig2/"

stnlist = f"../xlsx/ET_data.xlsx"
station_list = pd.read_excel(stnlist, header=0)  # ,header=0

os.chdir("%s" % (path))
for i in np.arange(len(station_list['filename'])):
    i = i + 10
    file = glob.glob(f"./{station_list['name'][i]}/ET_*D_*_LightGBM.nc")[0]
    ET_lgb = xr.open_dataset(file).ET
    ET_lgb = ET_lgb.where(ET_lgb.latitude >= -60, drop=True)
    N = len(ET_lgb.time)

    times = pd.date_range(f"{station_list['Byear'][i]}-01-01", f"{station_list['Eyear'][i]}-12-31", freq="1D")
    if i >= 9:
        times = []
        for year in np.arange(int(station_list['Byear'][i]), int(station_list['Eyear'][i]) + 1):
            mtimes = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="8D")
            times.append(mtimes)
        times = np.concatenate(times, axis=0)
    # In case ET data without time units, Eight-day data is produced separately

    ET = xr.open_dataset(f"/tera07/zhwei/For_QingChen/DataML/data/ET/{station_list['filename'][i]}").ET
    ET = ET.where(ET.latitude >= -60, drop=True)
    ET['time'] = times
    begin_year, end_year = station_list['T_Byear'][i], station_list['T_Eyear'][i]
    ET = ET.where((ET.time.dt.year >= begin_year) & (ET.time.dt.year <= end_year), drop=True)
    ET = ET[-N:]

    if station_list['map'][i] == 'remap':
        file = glob.glob(f"./{station_list['name'][i]}/metrics_*D_*_LightGBM.nc")[0]
        remap = xr.open_dataset(file).metric[1]
        ET_lgb = ET_lgb.where(remap >= -1, drop=True)
        ET = ET.where(remap >= -1, drop=True)

    if f"{station_list['name'][i]}" in models:
        ET_lgb = ET_lgb * 0.408
        ET = ET * 0.408
    ET_lgb = ET_lgb.mean(dim='time', skipna=True)
    ET = ET.mean(dim='time', skipna=True)
    # print(ET[100,:].values)
    # print(np.array(ET.transpose("longitude", "latitude"))[:,100])

    bins_4 = np.arange(0, 6.1, 0.1)
    nbin_4 = len(bins_4) - 1
    cmap4 = mpl.colormaps.get_cmap('Spectral_r')  # coolwarm/bwr/twilight_shifted
    norm4 = mpl.colors.BoundaryNorm(bins_4, nbin_4)
    locator = mpl.ticker.MultipleLocator(2)

    # Start with a square Figure.
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(1, 2)  # , left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05
    ax_map1 = fig.add_subplot(gs[0, 0])
    ax_map2 = fig.add_subplot(gs[0, 1])

    # map
    lon1, lat1 = ET_lgb.longitude.values, ET_lgb.latitude.values
    lon2, lat2 = ET.longitude.values, ET.latitude.values
    mmap1 = Basemap(projection='cyl', llcrnrlat=lat1[-1], urcrnrlat=lat1[0], llcrnrlon=lon1[0], urcrnrlon=lon1[-1], ax=ax_map1)
    mmap2 = Basemap(projection='cyl', llcrnrlat=lat2[-1], urcrnrlat=lat2[0], llcrnrlon=lon2[0], urcrnrlon=lon2[-1], ax=ax_map2)

    ilat1, ilon1 = np.meshgrid(lat1[::-1], lon1)
    ilat2, ilon2 = np.meshgrid(lat2[::-1], lon2)
    x1, y1 = mmap1(ilon1, ilat1)
    x2, y2 = mmap1(ilon2, ilat2)

    titles = ['Test', 'ET Product']
    ax_map1.set_title(titles[0])
    ax_map2.set_title(titles[1])

    C1 = mmap1.contourf(ilon1, ilat1, np.array(ET_lgb.transpose("longitude", "latitude"))[:, ::-1], levels=np.arange(0, 6.1, 0.1), alpha=0.8,
                        cmap=cmap4, norm=norm4)  #
    cb1 = mmap1.colorbar(C1, location='right', ticks=locator, size='3%')  # , extend='max', size='3%', pad='10%'
    cb1.set_label(label='ET [mm/day]', size=12)  # weight='bold'

    C2 = mmap2.contourf(ilon2, ilat2, np.array(ET.transpose("longitude", "latitude"))[:, ::-1], levels=np.arange(0, 6.1, 0.1), alpha=0.8,
                        cmap=cmap4, norm=norm4)
    cb2 = mmap2.colorbar(C2, location='right', ticks=locator, size='3%')  # , extend='max', size='3%', pad='10%'
    cb2.set_label(label='ET [mm/day]', size=12)  # weight='bold'

    mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=10, labels=[0, 0, 0, 1], color='silver')
    mmap1.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=10, labels=[1, 0, 0, 0], color='silver')
    mmap2.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=10, labels=[0, 0, 0, 1], color='silver')
    mmap2.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=10, labels=[1, 0, 0, 0], color='silver')
    mmap1.drawcoastlines(linewidth=0.5, color='black')
    mmap1.drawmapboundary()
    mmap2.drawcoastlines(linewidth=0.5, color='black')
    mmap2.drawmapboundary()

    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{pathout}/{station_list['name'][i]}.eps", dpi=200)
    plt.close()
    del ET, ET_lgb
    print(station_list['name'][i], 'done')
    exit(0)
