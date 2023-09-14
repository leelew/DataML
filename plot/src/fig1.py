import math
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import Normalize
from pylab import rcParams
from sklearn.metrics import r2_score, mean_squared_error

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
# models = ['FLUXCOM_9km', 'FLUXCOM']
# # path = "/tera07/zhwei/For_QingChen/DataML/output/forecast/"
'''
delete ET_2001-2016_EB_ET_1D_0p25.nc (done)
ET_3T/PLMV2 remap to delete sea area (done)
only 20% data of left (done)
FLUXCOM_9km,FLUXCOM transorm their units
'''
path = '/tera07/zhwei/For_QingChen/DataML/data/ET/'
pathout = "/tera07/zhwei/For_QingChen/DataML/plot/fig1/"
# # os.chdir("%s" % (path))
#
stnlist = f"{path}ET_data.xlsx"
station_list = pd.read_excel(stnlist, header=0)  # ,header=0

# split_ratio = 0.8
# ET = xr.Dataset()
# ET['time'] = pd.date_range(f"2012-01-01", f"2021-12-31", freq="1D")
#
# for i in np.arange(len(station_list['filename'])):
#     print(station_list['name'][i])
#     da = xr.open_dataset(f"{station_list['filename'][i]}").ET
#     if station_list['map'][i] == 'remap':
#         file = glob.glob(f"/tera07/zhwei/For_QingChen/DataML/output/forecast/{station_list['name'][i]}/metrics_*D_*_LightGBM.nc")[0]
#         remap = xr.open_dataset(file).metric[1]
#         da = da.where(remap > -1, drop=True)
#
#     # print(da.time)
#     N = int(split_ratio * len(da.time))  # only 20% data of left
#     da = np.nanmean(np.array(da), axis=(1, 2))[N:]
#     # da = da.groupby('time').mean(...)[N:]
#
#     times = pd.date_range(f"{station_list['Byear'][i]}-01-01", f"{station_list['Eyear'][i]}-12-31", freq="1D")
#     if i >= 9:
#         times = []
#         for year in np.arange(int(station_list['Byear'][i]), int(station_list['Eyear'][i]) + 1):
#             mtimes = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="8D")
#             times.append(mtimes)
#         times = np.concatenate(times, axis=0)
#     # In case ET data without time units, Eight-day data is produced separately
#
#     if f"{station_list['name'][i]}" in models:
#         da = da * 0.408
#     # Change FLUXCOM_9km/FLUXCOM data units to mm/day
#
#     et = xr.DataArray(da, coords={"time": times[N:]}, dims=["time"])
#     del da, times
#
#     ET[f"{station_list['name'][i]}"] = et
#
# ET.to_netcdf(f"{pathout}ET_last_20_percent.nc")
ET = xr.open_dataset(f"{pathout}ET_last_20_percent.nc")

fig, axes = plt.subplots(figsize=(28, 10))
colors = ['#6c4c49', '#FF7B89', '#8A5082', '#7EC636', '#758EB7', '#F58E6B', '#b2d235', '#33a3dc', '#bb505d', '#00a6ac', '#6950a1', '#dea32c']

for i in np.arange(len(station_list['filename'])):
    data = ET[f"{station_list['name'][i]}"]
    if i >= 9:
        data = data.where(data >= 0, drop=True)
    data.plot.line(x='time', linewidth=2, linestyle="--", alpha=0.8, label=f"{station_list['name'][i]}", color=colors[i])
axes.set_ylabel(f"ET Product [mm/day]", fontsize=18)
axes.set_yticks(np.around(np.arange(0.5, 3.1, 0.5), decimals=1), np.around(np.arange(0.5, 3.1, 0.5), decimals=1), fontsize=18)
axes.legend(loc='best', shadow=False, frameon=False, fontsize=18, ncol=4)
plt.tight_layout()
plt.savefig(f'{pathout}fig1_a.png', dpi=300)

# fig1 (b)----------------------------------------------------------------
'''
After Training, We have 12 sets of ET products with a time resolution of one day and a spatial resolution of 0.1 degrees 
'''

models_1 = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA']
models_2 = ['GLDAS_CLSM_2.2', 'GLDAS_Noah_2.1', 'ERA5', 'ET_3T']
models_3 = ['EB_ET', 'PMLV2', 'FLUXCOM', 'FLUXCOM_9km']

pathout = "/tera07/zhwei/For_QingChen/DataML/plot/fig1/"
fig, axes = plt.subplots(figsize=(28, 10))
color1 = ['#6c4c49', '#FF7B89', '#8A5082', '#7EC636']
color2 = ['#758EB7', '#F58E6B', '#b2d235', '#33a3dc']
color3 = ['#bb505d', '#00a6ac', '#6950a1', '#dea32c']

ds1 = xr.open_dataset('./ET_daily_1990-2020_cut.nc')
ds2 = xr.open_dataset('./ET_daily_1990-2020_cut_1.nc')
ds3 = xr.open_dataset('./ET_daily_1990-2020_cut_2.nc')
for i,model in enumerate(models_1):
    print(">>>", model)
    ds1[f"{model}"].plot.line(x='time', label=model, linewidth=1.5, linestyle="--", alpha=0.8, color=color1[i])
for i,model in enumerate(models_2):
    print(">>>", model)
    ds2[f"{model}"].plot.line(x='time', label=model, linewidth=1.5, linestyle="--", alpha=0.8, color=color2[i])
for i,model in enumerate(models_3):
    print(">>>", model)
    ds3[f"{model}"].plot.line(x='time', label=model, linewidth=1.5, linestyle="--", alpha=0.8, color=color3[i])
axes.legend(loc='best', shadow=False, frameon=False, fontsize=18, ncol=4)  # ,color=colors
axes.set_ylabel('ET [mm/day]', fontsize=18)
axes.set_yticks(np.around(np.arange(0.6, 2.8, 0.2), decimals=1), np.around(np.arange(0.6, 2.8, 0.2), decimals=1), fontsize=18)  # format="%.1f"
plt.tight_layout()
plt.savefig('%sfig1_b.png' % (pathout), dpi=300)
