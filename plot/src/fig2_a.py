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
from mpl_toolkits.basemap import Basemap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
### Plot settings
font = {'family': 'Times New Roman'}
# font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 16,
          'grid.linewidth': 0.3,
          'font.size': 15,
          'legend.fontsize': 13,
          'legend.frameon': False,
          'xtick.labelsize': 16,
          'xtick.direction': 'out',
          'ytick.labelsize': 16,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)

models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM_2.2', 'GLDAS_Noah_2.1', 'ERA5', 'ET_3T', 'EB_ET', 'PMLV2', 'FLUXCOM_9km',
          'FLUXCOM']
'''
ET_3T/PLMV2 remap to delete sea area (done)
only 20% data of left (done)
num:[r2, r, rmse]
'''

path = "/tera07/zhwei/For_QingChen/DataML/output/forecast/"
pathout = "/tera07/zhwei/For_QingChen/DataML/plot/fig2/"

stnlist = f"../xlsx/ET_data.xlsx"
station_list = pd.read_excel(stnlist, header=0)  # ,header=0

# remap_0p1 = xr.open_dataset("/tera07/zhwei/For_QingChen/DataML/data/ET/ET_2001-2016_EB_ET_1D_0p1.nc").ET[3]
# remap_0p25 = xr.open_dataset("/tera07/zhwei/For_QingChen/DataML/data/ET/ET_2000-2020_GLDAS_Noah_2.1_1D_0p25.nc").ET[0]
# split_ratio = 0.8

os.chdir("%s" % (path))
for i in np.arange(len(station_list['filename'])):
    file = glob.glob(f"./{station_list['name'][i]}/metrics_*D_*_LightGBM.nc")[0]
    print(file)
    metric = xr.open_dataset(file).metric
    metric = metric.where(metric.latitude >= -60, drop=True)

    if station_list['map'][i] == 'remap':
        metric = metric.where(metric[1] >= -1, drop=True)

    r2 = metric[0]
    rmse = metric[2]


    bins_4 = np.arange(0, 1.1, 0.01)
    nbin_4 = len(bins_4) - 1
    cmap4 = mpl.cm.get_cmap('Spectral_r', nbin_4)  # coolwarm/bwr/twilight_shifted
    # cmap4 = mpl.colormaps.get_cmap('Spectral_r')  # coolwarm/bwr/twilight_shifted
    norm4 = mpl.colors.BoundaryNorm(bins_4, nbin_4)
    locator = mpl.ticker.MultipleLocator(0.2)

    # Start with a square Figure.
    fig = plt.figure(figsize=(15, 10))
    ax_map1 = fig.add_subplot()


    # map
    lon1, lat1 = metric.longitude.values, metric.latitude.values
    mmap1 = Basemap(projection='cyl', llcrnrlat=lat1[-1], urcrnrlat=lat1[0], llcrnrlon=lon1[0], urcrnrlon=lon1[-1], ax=ax_map1)

    ilat1, ilon1 = np.meshgrid(lat1[::-1], lon1)
    print(lon1)
    print(ilon1)


    x1, y1 = mmap1(ilon1, ilat1)

    titles = ['R2']
    ax_map1.set_title(titles[0])

    C1 = mmap1.contourf(ilon1, ilat1, np.array(r2.transpose("longitude", "latitude"))[:, ::-1], levels=np.arange(0, 1.1, 0.01), alpha=0.8,
                        cmap=cmap4, norm=norm4)  #
    cb1 = mmap1.colorbar(C1, location='right', ticks=locator,  size='3%')  # , extend='max', size='3%', pad='10%'
    cb1.set_label(label='R2', size=12)  # weight='bold'


    mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=10, labels=[0, 0, 0, 1], color='silver')
    mmap1.drawparallels(np.arange(90, -60, -30), linewidth=0.5, fontsize=10, labels=[1, 0, 0, 0], color='silver')
    mmap1.drawcoastlines(linewidth=0.5, color='black')
    mmap1.drawmapboundary()


    plt.tight_layout()
    plt.draw()
    plt.savefig(f"{pathout}/{station_list['name'][i]}_R2.eps", dpi=200)
    plt.close()
    exit(0)

    # fig = plt.figure(figsize=(15, 10))
    # ax_map2 = fig.add_subplot()
    # 
    # # map
    # lon2, lat2 = metric.longitude.values, metric.latitude.values
    # mmap2 = Basemap(projection='cyl', llcrnrlat=lat2[-1], urcrnrlat=lat2[0], llcrnrlon=lon2[0], urcrnrlon=lon2[-1], ax=ax_map2)
    #
    # ilat2, ilon2 = np.meshgrid(lat2[::-1], lon2)
    # x2, y2 = mmap2(ilon2, ilat2)
    #
    # titles = ['RMSE']
    # ax_map2.set_title(titles[0])
    #
    # C2 = mmap2.contourf(ilon2, ilat2, np.array(rmse.transpose("longitude", "latitude"))[:, ::-1], levels=np.arange(0, 1.1, 0.01), alpha=0.8,
    #                     cmap=cmap4, norm=norm4)
    # cb2 = mmap2.colorbar(C2, location='right', ticks=locator,  size='3%')  # , extend='max', size='3%', pad='10%'
    # cb2.set_label(label='RMSE', size=12)  # weight='bold'
    #
    # mmap2.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=10, labels=[0, 0, 0, 1], color='silver')
    # mmap2.drawparallels(np.arange(90, -60, -30), linewidth=0.5, fontsize=10, labels=[1, 0, 0, 0], color='silver')
    # mmap2.drawcoastlines(linewidth=0.5, color='black')
    # mmap2.drawmapboundary()
    #
    # plt.tight_layout()
    # plt.draw()
    # plt.savefig(f"{pathout}/{station_list['name'][i]}_RMSE.png", dpi=200)
    # plt.close()
    # print(station_list['name'][i], 'done')



    # bins_4 = np.arange(0, 1.01, 0.01)
    # nbin_4 = len(bins_4) - 1
    # cmap4 = mpl.cm.get_cmap('Spectral_r', nbin_4)  # coolwarm/bwr/twilight_shifted
    # norm4 = mpl.colors.BoundaryNorm(bins_4, nbin_4)
    # locator = mpl.ticker.MultipleLocator(0.1)
    #
    # fig = plt.figure(figsize=(15, 10))  # figsize=(5, 5)
    # ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # r2.plot.pcolormesh(x="longitude", y="latitude", cmap=cmap4, norm=norm4, robust=True, cbar_kwargs={"ticks": locator, "shrink": 0.6},
    #                    transform=ccrs.PlateCarree())  # "label": "R2",
    # ax1.coastlines()
    # ax1.gridlines(draw_labels=False, linestyle=':', linewidth=0.3, color='k')
    # ax1.set_extent([-180, 180, -60, 90])
    # ax1.set_xticks(np.arange(-150, 180, 30), crs=ccrs.PlateCarree())
    # ax1.set_yticks([-60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter()
    # lat_formatter = LatitudeFormatter()
    # ax1.xaxis.set_major_formatter(lon_formatter)
    # ax1.yaxis.set_major_formatter(lat_formatter)
    # ax1.set_title(f"R2", fontsize=16)
    # plt.tick_params(labelsize=9)
    # plt.tight_layout()
    # plt.draw()
    # plt.savefig(f"{pathout}/{station_list['name'][i]}_r2.png", dpi=200)
    # plt.close()
    #
    # fig = plt.figure(figsize=(15, 10))  # figsize=(5, 5)
    # ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # rmse.plot.pcolormesh(x="longitude", y="latitude", cmap=cmap4, norm=norm4, robust=True,
    #                      cbar_kwargs={"ticks": locator, "shrink": 0.6},
    #                      transform=ccrs.PlateCarree())  # "label": "RMSE",
    # ax2.add_feature(cfeature.COASTLINE)
    # ax2.gridlines(draw_labels=False, linestyle=':', linewidth=0.3, color='k')
    # ax1.set_extent([-180, 180, -60, 90])
    # ax1.set_xticks(np.arange(-150, 180, 30), crs=ccrs.PlateCarree())
    # ax1.set_yticks([-60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    # # ax2.set_xticks([-90, 0, 90], crs=ccrs.PlateCarree())
    # # ax2.set_yticks([-45, 0, 45], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter()
    # lat_formatter = LatitudeFormatter()
    # ax2.xaxis.set_major_formatter(lon_formatter)
    # ax2.yaxis.set_major_formatter(lat_formatter)
    # ax2.set_title(f"RMSE", fontsize=16)
    # plt.tick_params(labelsize=9)
    # plt.tight_layout()
    # plt.draw()
    # plt.savefig(f"{pathout}/{station_list['name'][i]}_RMSE.png", dpi=200)
    # plt.close()
