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
# from mpl_toolkits.basemap import Basemap
import argparse
from config import get_site_args
from pathlib import PosixPath, Path
from tqdm import tqdm, trange

# from sklearn.model_selection import train_test_split


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


def filter_nan(s=np.array([]), o=np.array([])):
    """
    this functions removed the data from simulated and observed data
    whereever the observed data contains nan

    this is used by all other functions, otherwise they will produce nan as
    output
    """
    data = np.array([s.flatten(), o.flatten()])
    data = np.transpose(data)
    data = data[~np.isnan(data).any(1)]

    return data[:, 0], data[:, 1]


def R2(o, s):
    """
    correlation coefficient R2
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """
    if s.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(o, s)[0, 1]
    return corr ** 2


def rmse(y_true, y_pred):
    """
    计算RMSE分数
    """
    mse = np.mean((y_pred - y_true) ** 2)
    RMSE = np.sqrt(mse)
    return RMSE


def correlation(o, s):
    """
    correlation coefficient
    input:
        s: simulated
        o: observed
    output:
        correlation: correlation coefficient
    """

    corr = np.corrcoef(o, s)[0, 1]

    return corr


def KGE(y_true, y_pred):
    """
    Kling-Gupta Efficiency
    input:
        y_pred: simulated
        y_true: observed
    output:
        kge: Kling-Gupta Efficiency
        r: correlation
        std_ratio: ratio of the standard deviation
        bias_ratio: ratio of the mean
    """
    r = np.corrcoef(y_true, y_pred)[0, 1]
    std_ratio = np.std(y_pred) / np.std(y_true)
    bias_ratio = np.mean(y_pred) / np.mean(y_true)
    KGE = 1 - np.sqrt((r - 1) ** 2 + (std_ratio - 1) ** 2 + (bias_ratio - 1) ** 2)
    return KGE


def plot_validation_metric_test(dir_fig, gauge_lon, gauge_lat, metric, cmap, norm, ticks, var):
    fig = plt.figure()
    # fig.suptitle("f_lfevpa (JULES-CoLM2014)", fontsize=10, y=0.75)

    # fig.set_tight_layout(True)
    M = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l')
    # M = Basemap(projection='robin', resolution='l', lat_0=15, lon_0=0)
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)
    M.drawcountries(color='0.6', linewidth=0.1)
    M.drawparallels(np.arange(-60., 60., 30.), dashes=[1, 1], linewidth=0.25, color='0.5')
    M.drawmeridians(np.arange(0., 360., 60.), dashes=[1, 1], linewidth=0.25, color='0.5')

    loc_lon, loc_lat = M(gauge_lon, gauge_lat)
    cs = M.scatter(loc_lon, loc_lat, 50, metric, cmap=cmap, norm=norm, marker='.', edgecolors='none', alpha=0.9)
    # cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
    cbaxes = fig.add_axes([0.45, 0.32, 0.35, 0.012])

    cb = fig.colorbar(cs, cax=cbaxes, ticks=ticks, orientation='horizontal', spacing='uniform')
    cb.solids.set_edgecolor("face")
    # cb.set_label('KGE change', position=(0.5, 1.5), labelpad=-35)
    cb.set_label('%s' % (var), position=(0.50, 1.5), labelpad=-21.5, fontsize=8)
    cb.ax.tick_params(labelsize=6)
    cb.ax.tick_params(length=2)
    cb.ax.tick_params(pad=1)
    # cb.set_label('log10(RMSE)', position=(0.5, 1.5), labelpad=-35)
    # cb.set_label('PBIAS', position=(0.5, 1.5), labelpad=-35)

    plt.savefig('/stu01/baif/CoLM202X/colm_single_test/code/validation/cases/day_0901/le/metrics/reduce/7/f_lfevpa_%s.png' % (var), format='png',
                dpi=500)
    # plt.show()


# def plot_validation_metric(pathout, ilon_idx, ilat_idx, metric, y_bottom, y_top, cfg):
#     Metrics = ['R2', 'R', 'RMSE']
#     # Metrics = ['R2', 'MSE', 'RMSE', 'KGE']
#     for i in range(metric.shape[0]):
#         fig = plt.figure(figsize=(14, 10))  # figsize=(5, 5)
#         ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#
#         # bins = np.arange(y_bottom[i], y_top[i], 0.1)
#         # nbin = len(bins) - 1
#         # norm_idx = mpl.colors.BoundaryNorm(bins, nbin)
#         locator = mpl.ticker.MultipleLocator(0.2)
#
#         site = ax.scatter(ilon_idx, ilat_idx, s=1, c=metric[i], cmap='coolwarm', marker='o')  # , norm=norm_idx
#         # ax.legend(title=f"N = {metric.shape[1]}", loc='lower left', shadow=False, frameon=False, fontsize=15, ncol=2, title_fontsize=20)
#         cbar = plt.colorbar(site, ax=ax, location='right', ticks=locator)
#         # ax.colorbar(site, location='right', ticks=locator, size='3%')  # , extend='max', size='3%', pad='10%'
#         ax.add_feature(cfeature.LAND)  # , facecolor='brown'
#         # ax.add_feature(cfeature.COASTLINE)
#         # ax.add_feature(cfeature.OCEAN)
#         # ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.3, color='k')
#
#         ax.set_extent([30, 140, -60, -20])
#         ax.set_xticks(np.arange(30, 140 + 30, 30), crs=ccrs.PlateCarree())
#         ax.set_yticks(np.arange(-60, -20 + 30, 30), crs=ccrs.PlateCarree())
#         lon_formatter = LongitudeFormatter()
#         lat_formatter = LatitudeFormatter()
#         ax.xaxis.set_major_formatter(lon_formatter)
#         ax.yaxis.set_major_formatter(lat_formatter)
#
#         # plt.tight_layout()
#         plt.savefig(f"{pathout}/fig2_{cfg['et_product']}_sitetest_{Metrics[i]}.png", dpi=300)
#         # plt.savefig(f"{pathout}/fig2_{cfg['et_product']}_sitetest.eps", dpi=300)

def plot_validation_metric(pathout, ilon_idx, ilat_idx, y_bottom, y_top, cfg):
    Metrics = ['R2', 'R', 'RMSE']
    Metrics_data = pd.read_excel(f"{pathout}/idx/{cfg['et_product']}_mapdata.xlsx", header=0, sheet_name=f'test')
    for i, metric in enumerate(Metrics):
        fig = plt.figure(figsize=(14, 10))  # figsize=(5, 5)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # sns.scatterplot(data=Metrics_data, x="lon", y="lat", sizes=5, palette=sns.color_palette("coolwarm", n_colors=30, desat=.5),
        #                 edgecolor='black',
        #                 hue=f"{metric}",
        #                 alpha=1)  # ,marker=markers
        ax.scatter(Metrics_data['lon'], Metrics_data['lat'], s=1, c=Metrics_data[f'{metric}'], cmap='coolwarm', marker='o')
        # ax.legend(title=f"N = {len(station_list['filename'])}", loc='lower left', shadow=False, frameon=False, fontsize=15, ncol=2, title_fontsize=20)

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

        # plt.tight_layout()
        plt.savefig(f"{pathout}/fig2_{cfg['et_product']}_sitetest_{Metrics[i]}.png", dpi=300)
        # plt.savefig(f"{pathout}/fig2_{cfg['et_product']}_sitetest.eps", dpi=300)


if __name__ == "__main__":
    cfg = get_site_args()
    model = cfg['et_product']
    tr = cfg['temporal_resolution']
    sr = cfg['spatial_resolution']
    input_path = cfg['inputs_path']
    out_path = cfg['outputs_path']
    pathout = "/tera07/zhwei/For_QingChen/DataML/plot/fig2/"
    path = cfg["inputs_path"] + cfg["et_product"] + '/'
    if os.path.exists(pathout + f'idx/{model}_ilat_idx1.npy'):
        ilat_idx, ilon_idx = np.load(pathout + f'idx/{model}_ilat_idx.npy'), np.load(pathout + f'idx/{model}_ilon_idx.npy')
    else:
        lat_file_name = 'lat_{tr}_{sr}.npy'.format(tr=cfg["temporal_resolution"], sr=cfg["spatial_resolution"])
        lon_file_name = 'lon_{tr}_{sr}.npy'.format(tr=cfg["temporal_resolution"], sr=cfg["spatial_resolution"])
        lat, lon = np.load(path + lat_file_name), np.load(path + lon_file_name)
        # ilat, ilon = np.meshgrid(lat, lon)
        ilon, ilat = np.meshgrid(lon, lat)
        ilat = ilat.reshape(-1)[np.newaxis, :]
        ilon = ilon.reshape(-1)[np.newaxis, :]

        file_name = f"{model}_ET_{tr}_{sr}_{cfg['begin_year']}_{cfg['end_year']}.npy"
        et = np.load(path + file_name)  # (t,lat,lon)

        nt, nlat, nlon = et.shape
        et = et.reshape(nt, -1)
        ilat = np.delete(ilat, np.argwhere(np.isnan(et)), axis=1)
        ilon = np.delete(ilon, np.argwhere(np.isnan(et)), axis=1)
        # et = np.delete(et, np.argwhere(np.isnan(et)), axis=1)
        print(ilat.shape, ilon.shape, et.shape)
        # exit(0)
        test_idx = np.load(path + 'test_idx.npy')
        ilat_idx = ilat[:, test_idx]
        ilon_idx = ilon[:, test_idx]
        np.save(f'{model}_ilat_idx.npy', ilat_idx)
        np.save(f'{model}_ilon_idx.npy', ilon_idx)
        os.system('mv {} {}'.format("*.npy", pathout + 'idx'))

    # print(ilon_idx[0][0],ilon_idx[0][-1],ilat_idx[0][0],ilat_idx[0][-1])
    # y_test = np.load(path + 'y_sitetest.npy')[:, :, 0]
    # name = f"ET_{model}_{tr}_{sr}_{cfg['begin_year']}_{cfg['end_year']}_sitetest_LightGBM.nc"
    # y_pred = np.array(xr.open_dataset(out_path + f"/forecast/{model}/{name}").ET)
    # test_point = y_test.shape[1]
    # data = np.full((4, test_point), np.nan)
    # pbar = tqdm(range(test_point), ncols=140)
    # for i in pbar:
    #     pbar.set_description("Now at %s" % (i + 1))
    #     o, s = y_test[:, i], y_pred[:, i]
    #     data[:, i] = [R2(o, s), correlation(o, s), rmse(o, s), KGE(o, s)]

    data = xr.open_dataset(
        out_path + f"/forecast/{model}/metrics_{model}_{tr}_{sr}_{cfg['begin_year']}_{cfg['end_year']}_sitetest_LightGBM.nc").metric
    y_bottom = [0, 0, -0.5]
    y_top = [1, 2, 1]
    # y_bottom = [0, 0, 0, -0.5]
    # y_top = [1, 2, 2, 1]
    print(ilat_idx[0].shape)

    map_data = pd.DataFrame({'i': range(ilat_idx.shape[1]),
                             'lat': ilat_idx[0],
                             'lon': ilon_idx[0],
                             'R2': data[0].values,
                             'R': data[1].values,
                             'RMSE': data[2].values})  # 'R': R,'KGE': KGE,
    map_data.to_excel(f"{pathout}/idx/{model}_mapdata.xlsx", sheet_name='test', index=True)

    plot_validation_metric(pathout, ilon_idx[0], ilat_idx[0], y_bottom, y_top, cfg)
