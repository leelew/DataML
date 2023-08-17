import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import scipy
from scipy import stats
from statistics import mean
from pylab import rcParams
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

### Plot settings
font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)
params = {'backend': 'ps',
          'axes.labelsize': 16,
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

stnlist = f"../xlsx/case5_model2.xlsx"
station_list = pd.read_excel(stnlist, header=0, sheet_name='train_test')  # ,header=0

fig = plt.figure(figsize=(14, 10))  # figsize=(5, 5)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

sns.scatterplot(data=station_list, x="lon", y="lat", sizes=25, palette=sns.color_palette("Set3", n_colors=15, desat=.5), edgecolor='black',
                hue="Climate_zone",
                alpha=1)  # ,marker=markers
ax.scatter(station_list['lon'], station_list['lat'], s=1, color='black')
ax.legend(title=f"N = {len(station_list['filename'])}", loc='lower left', shadow=False, frameon=False, fontsize=15, ncol=2, title_fontsize=20)

ax.add_feature(cfeature.LAND)  # , facecolor='brown'
ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.3, color='k')

ax.set_extent([-180, 180, -60, 90])
ax.set_xticks(np.arange(-180, 180 + 30, 30), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-60, 90 + 30, 30), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

plt.tight_layout()
plt.savefig(f"../fig4/fig4_In_situ.png", dpi=300)
plt.savefig(f"../fig4/fig4_In_situ.eps", dpi=300)
# plt.show()

stnlist = f"../xlsx/case5_model2.xlsx"
station_list = pd.read_excel(stnlist, header=0, sheet_name='train_test')  # ,header=0

fig = plt.figure(figsize=(5, 5))  # figsize=(5, 5)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

sns.scatterplot(data=station_list, x="lon", y="lat", sizes=25, palette=sns.color_palette("Set3", n_colors=15, desat=.5), edgecolor='black',
                hue="Climate_zone",
                alpha=1)  # ,marker=markers
ax.scatter(station_list['lon'], station_list['lat'], s=1, color='black')
ax.legend(title=f"N = {len(station_list['filename'])}", loc='lower left', shadow=False, frameon=False, fontsize=5, ncol=2, title_fontsize=5)

ax.add_feature(cfeature.LAND)  # , facecolor='brown'
ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.OCEAN)
ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.3, color='k')

ax.set_extent([-10, 25, 35, 60])
ax.set_xticks(np.arange(-10, 25 + 5, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(35, 60 + 5, 5), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

plt.tight_layout()
plt.savefig(f"../fig4/fig4_In_situ1.png", dpi=300)
plt.savefig(f"../fig4/fig4_In_situ1.eps", dpi=300)