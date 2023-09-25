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
import seaborn as sns

font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 25,
          'grid.linewidth': 0.2,
          'font.size': 20,
          'legend.fontsize': 20,
          'legend.frameon': False,
          'xtick.labelsize': 20,
          'xtick.direction': 'out',
          'ytick.labelsize': 20,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def plot_ancillary():
    path = "/tera07/zhwei/For_QingChen/DataML/forecast/input/ancillary/0p1/"
    pathout = "/tera07/zhwei/For_QingChen/DataML/plot/ancillary/"

    os.chdir("%s" % (path))
    filenames = sorted(glob.glob('*.nc'))
    levels = [[0, 31, 1, 5], [-350, 6800, 100, 1000], [0, 20.2, 0.2, 2], [0, 1.005, 0.005, 0.2], [100, 1810, 10, 200]]
    level_add = [[0, 3.55, 0.05, 0.5], [0, 101, 1, 10]]
    for i, filename in enumerate(filenames):
        print('>>>>', filename[:-3])
        ancillary = xr.open_dataset(filename).climate_zone
        ancillary = ancillary.where(ancillary.latitude >= -60, drop=True)
        lon, lat = ancillary.longitude.values, ancillary.latitude.values
        if filename == 'LC_0p1.nc':
            ancillary = ancillary.where(ancillary > 0, np.nan)

        if 'soilgrid' in filename:
            for j in [0, 1, 7, 8, 14, 15]:
                print('>>> soilgrid', j)
                fig = plt.figure(figsize=(15, 10))
                ax_map = fig.add_subplot(111)
                mmap1 = Basemap(projection='cyl', llcrnrlat=lat[-1], urcrnrlat=lat[0], llcrnrlon=lon[0], urcrnrlon=lon[-1], ax=ax_map)
                ilon, ilat = np.meshgrid(lon, lat)
                x1, y1 = mmap1(ilon, ilat)
                if j < 7:
                    level = levels[i]
                if j >= 7:
                    level = level_add[1]
                locator = mpl.ticker.MultipleLocator(level[3])
                print(ancillary[j].max().values, ancillary[j].min().values)
                C1 = mmap1.contourf(ilon, ilat, np.array(ancillary[j]), levels=np.arange(level[0], level[1], level[2]), alpha=1, cmap='Spectral_r')  #
                cb1 = mmap1.colorbar(C1, location='right', ticks=locator, size='3%', extend='both')  # , extend='max', size='3%', pad='10%'
                cb1.set_label(label='Ancillary', size=12)  # weight='bold'
                ax_map.set_xlabel("Longitude", fontsize=25, labelpad=25)
                ax_map.set_ylabel("Latitude", fontsize=25, labelpad=50)
                mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=20, labels=[0, 0, 0, 1], color='silver')
                mmap1.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=20, labels=[1, 0, 0, 0], color='silver')
                mmap1.drawcoastlines(linewidth=0.5, color='black')
                mmap1.drawmapboundary()
                plt.draw()
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.png", dpi=200)
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.pdf", dpi=200)
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.eps", dpi=200)
                plt.close(fig)
            break
        if 'kosugi' in filename:
            for j in range(ancillary.shape[0]):
                print('>>> kosugi', j)
                fig = plt.figure(figsize=(15, 10))
                ax_map = fig.add_subplot(111)
                mmap1 = Basemap(projection='cyl', llcrnrlat=lat[-1], urcrnrlat=lat[0], llcrnrlon=lon[0], urcrnrlon=lon[-1], ax=ax_map)
                ilon, ilat = np.meshgrid(lon, lat)
                x1, y1 = mmap1(ilon, ilat)

                level = levels[i]
                if (j == 1) | (j == 2):
                    level = level_add[0]
                print(ancillary[j].max().values, ancillary[j].min().values)

                locator = mpl.ticker.MultipleLocator(level[3])
                C1 = mmap1.contourf(ilon, ilat, np.array(ancillary[j]), levels=np.arange(level[0], level[1], level[2]), alpha=1, cmap='Spectral_r')  #
                cb1 = mmap1.colorbar(C1, location='right', ticks=locator, size='3%', extend='both')  # , extend='max', size='3%', pad='10%'
                cb1.set_label(label='Ancillary', size=12)  # weight='bold'
                ax_map.set_xlabel("Longitude", fontsize=25, labelpad=25)
                ax_map.set_ylabel("Latitude", fontsize=25, labelpad=50)
                mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=20, labels=[0, 0, 0, 1], color='silver')
                mmap1.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=20, labels=[1, 0, 0, 0], color='silver')
                mmap1.drawcoastlines(linewidth=0.5, color='black')
                mmap1.drawmapboundary()
                plt.draw()
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.png", dpi=200)
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.pdf", dpi=200)
                plt.savefig(f"{pathout}/{filename[:-3]}_{j}.eps", dpi=200)
                plt.close(fig)
        else:
            print(f'>>> {filename[:-3]} 3')
            fig = plt.figure(figsize=(15, 10))
            ax_map = fig.add_subplot(111)
            mmap1 = Basemap(projection='cyl', llcrnrlat=lat[-1], urcrnrlat=lat[0], llcrnrlon=lon[0], urcrnrlon=lon[-1], ax=ax_map)
            ilon, ilat = np.meshgrid(lon, lat)
            x1, y1 = mmap1(ilon, ilat)

            level = levels[i]
            locator = mpl.ticker.MultipleLocator(level[3])
            C1 = mmap1.contourf(ilon, ilat, np.array(ancillary), levels=np.arange(level[0], level[1], level[2]), alpha=1, cmap='Spectral_r')  #
            cb1 = mmap1.colorbar(C1, location='right', ticks=locator, size='3%')  # , extend='max', size='3%', pad='10%'
            cb1.set_label(label='Ancillary', size=12)  # weight='bold'
            ax_map.set_xlabel("Longitude", fontsize=25, labelpad=25)
            ax_map.set_ylabel("Latitude", fontsize=25, labelpad=50)
            mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=20, labels=[0, 0, 0, 1], color='silver')
            mmap1.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=20, labels=[1, 0, 0, 0], color='silver')
            mmap1.drawcoastlines(linewidth=0.5, color='black')
            mmap1.drawmapboundary()
            plt.draw()
            plt.savefig(f"{pathout}/{filename[:-3]}.png", dpi=200)
            plt.savefig(f"{pathout}/{filename[:-3]}.pdf", dpi=200)
            plt.savefig(f"{pathout}/{filename[:-3]}.eps", dpi=200)
            plt.close()
            exit(0)

    print('Ancillary Done')


def plot_LAI():
    path = "/tera07/zhwei/For_QingChen/DataML/forecast/input/LAI/1D_0p1/"
    pathout = "/tera07/zhwei/For_QingChen/DataML/plot/ancillary/"

    os.chdir("%s" % (path))
    filename = sorted(glob.glob('*.nc'))[0]
    print('>>>>', filename[:-3])
    mask = xr.open_dataset('/tera07/zhwei/For_QingChen/DataML/forecast/input/ancillary/0p1/Beck_KG_V1_present_0p1.nc').climate_zone

    lai = xr.open_dataset(filename).lai
    lai = lai.where(mask >= 0, np.nan)
    lai = lai.where(lai.latitude >= -60, drop=True)
    lon, lat = lai.longitude.values, lai.latitude.values
    lai_sr = np.mean(np.array(lai.values), axis=0)
    lai_tr = np.nanmean(np.array(lai.values), axis=(1, 2))

    print(f'>>> plot area')
    fig1 = plt.figure(figsize=(13, 10))
    ax_map = fig1.add_subplot(111)
    mmap1 = Basemap(projection='cyl', llcrnrlat=lat[-1], urcrnrlat=lat[0], llcrnrlon=lon[0], urcrnrlon=lon[-1], ax=ax_map)
    ilon, ilat = np.meshgrid(lon, lat)
    x1, y1 = mmap1(ilon, ilat)

    locator = mpl.ticker.MultipleLocator(2)
    C1 = mmap1.contourf(ilon, ilat, np.array(lai_sr), levels=np.arange(0, 10.1, 0.1), alpha=1, cmap='YlGn')  #
    cb1 = mmap1.colorbar(C1, location='right', ticks=locator, size='3%', extend='both')  # , extend='max', size='3%', pad='10%'
    cb1.set_label(label='LAI', size=12)  # weight='bold'
    ax_map.set_xlabel("Longitude", fontsize=25, labelpad=25)
    ax_map.set_ylabel("Latitude", fontsize=25, labelpad=50)
    mmap1.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, fontsize=20, labels=[0, 0, 0, 1], color='silver')
    mmap1.drawparallels(np.arange(90, -90, -30), linewidth=0.5, fontsize=20, labels=[1, 0, 0, 0], color='silver')
    mmap1.drawcoastlines(linewidth=0.5, color='black')
    mmap1.drawmapboundary()
    plt.draw()
    plt.savefig(f"{pathout}/{filename[:-3]}_area.eps", dpi=200)
    plt.savefig(f"{pathout}/{filename[:-3]}_area.png", dpi=200)
    plt.savefig(f"{pathout}/{filename[:-3]}_area.pdf", dpi=200)
    plt.close(fig1)

    # print(f'>>> plot line')
    # fig2 = plt.figure(figsize=(15, 6))
    # ax_line = fig2.add_subplot(111)
    # colors = sns.color_palette("Set3", n_colors=2, desat=.9).as_hex()
    # time = pd.date_range('2000-01-01', '2000-12-31', freq='1D')
    # # xtime = xr.DataArray(time, coords={"time": time}, dims=["time"])
    # # lai_plot = xr.DataArray(lai_tr, coords={"time": time}, dims=["time"])
    # # print(lai_tr)
    # # lai_plot.plot.line(x='time', ax=ax_line, color=colors[0], lw=1.5, linestyle="--")  # --
    # ax_line.plot(lai_tr, color=colors[0], lw=1.5, linestyle="--")
    #
    # X = range(len(time))
    # xtime = xr.DataArray(time, coords={"time": time}, dims=["time"])
    # idx = xr.where(xtime.time.dt.day==1, X, np.nan)
    # idxs = np.delete(np.array(idx.values),np.argwhere(np.isnan(idx.values)),axis=0)
    # Xlabel = [f"{xtime[int(i)].values}" for i in idxs]
    # Xlabels = [f"{label[5: -19]}" for label in Xlabel]
    # ax_line.set_xticks(idxs,Xlabels, fontsize=30,rotation=35,)
    # ax_line.set_yticks(np.arange(0.5, 2.5, .5), fontsize=35)
    # plt.draw()
    # plt.savefig(f"{pathout}/{filename[:-3]}_line.png", dpi=200)
    # plt.savefig(f"{pathout}/{filename[:-3]}_line.pdf", dpi=200)
    # plt.savefig(f"{pathout}/{filename[:-3]}_line.eps", dpi=200)
    # plt.close(fig2)


if __name__ == "__main__":


    # plot_LAI()
    plot_ancillary()