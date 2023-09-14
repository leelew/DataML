import math
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import glob, os, shutil, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
from pylab import rcParams
from tqdm import tqdm, trange
from matplotlib import colors
from sklearn.metrics import r2_score, mean_squared_error
import argparse
from config import get_args
from pathlib import PosixPath, Path


### Plot settings
font = {'family': 'Times new roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 0.2,
          'font.size': 20,
          'legend.fontsize': 20,
          'legend.frameon': False,
          'xtick.labelsize': 32,
          'xtick.direction': 'out',
          'ytick.labelsize': 35,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def box_plot(cfg, y_lim, y):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
              'PMLV2']
    ML_models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    ET_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_ET_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')
    ET_data = ET_data.dropna(subset=["data"])
    ML_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')

    Metric = ['R2', 'MSE', 'RMSE', 'KGE']

    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    position = np.arange(1, 2 * (len(models) + len(ML_models)), 2)
    colors = sns.color_palette("Set3", n_colors=(len(models) + len(ML_models)), desat=.7).as_hex()

    for i, mm in enumerate(Metric):
        Metric_data1 = ET_data[ET_data['Metric'].isin([mm])]
        data = []
        for model in (models):
            data.append(Metric_data1[Metric_data1['models'].isin([model])].data)
        Metric_data2 = ML_data[ML_data['Metric'].isin([mm])]
        for model in (ML_models):
            data.append(Metric_data2[Metric_data2['models'].isin([model])].data)
        bplot = axes[i].boxplot(data, patch_artist=True, positions=position,
                                widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                capprops={"color": "black", "linewidth": 2.0},
                                whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for j, model in zip(np.arange(0, len(models)), models):
            df = Metric_data1[Metric_data1['models'].isin([model])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')
        for j, model in zip(np.arange(len(models), len(models)+len(ML_models)), ML_models):
            df = Metric_data2[Metric_data2['models'].isin([model])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')


        # axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=Metric_data2[Metric_data2['models'].isin(['AutoML'])].data.median(), ls="--", c="red", alpha=0.7)  # 添加水平直线 #105885
        axes[i].axvline(x=len(models)*2, ls="--", c="black", alpha=0.7)  # x=xposition[len(compare_lists)]-1
        axes[i].set(ylim=y_lim[i])
        axes[i].set_ylabel(f'{mm}', fontsize=40, )

    position1 = np.append(position, values=position[-1] + 2)
    axes[-1].set_xticks([i for i in position1], models+ML_models+[''], rotation=35, fontsize=30,weight='bold')
    # fig.legend(bplot['boxes'], models, loc=8, borderaxespad=2, ncol=4, shadow=False, fontsize=35)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig5/fig5_{cfg['case_name']}_{cfg['model_name']}_{cfg['testset']}.png", dpi=300)
    print(f"fig5: {cfg['case_name']} {cfg['model_name']} {cfg['testset']} Done -----")


def box_plot_select(cfg, y_lim, y):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    models = cfg['models']
    # models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'ERA5', 'REA', 'GLEAM_hybrid', 'FLUXCOM_9km', 'DNN', 'LightGBM', 'Random Forest']
    ML_models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    ET_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}_ET.xlsx", header=0, sheet_name=f'{test_set}')
    ML_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')

    Metric = ['R2', 'MSE', 'RMSE', 'KGE']

    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    position = np.arange(1, 2 * (len(models) + len(ML_models)), 2)
    colors = sns.color_palette("Set3", n_colors=len(Metric), desat=.7).as_hex()

    for i, mm in enumerate(Metric):
        Metric_data1 = ET_data[ET_data['Metric'].isin([mm])]
        data = []
        for model in (models):
            data.append(Metric_data1[Metric_data1['models'].isin([model])].data)
        Metric_data2 = ML_data[ML_data['Metric'].isin([mm])]
        for model in (ML_models):
            data.append(Metric_data2[Metric_data2['models'].isin([model])].data)
        bplot = axes[i].boxplot(data, patch_artist=True, positions=position,
                                widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                capprops={"color": "black", "linewidth": 2.0},
                                whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for j, model in zip(np.arange(0, len(models)), models):
            df = Metric_data1[Metric_data1['models'].isin([model])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')
        for j, model in zip(np.arange(len(models), len(models)+len(ML_models)), ML_models):
            df = Metric_data2[Metric_data2['models'].isin([model])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')


        # axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=Metric_data2[Metric_data2['models'].isin(['AutoML'])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=y_lim[i])
        axes[i].set_ylabel(f'{mm}', fontsize=40, )

    position1 = np.append(position, values=position[-1] + 2)
    axes[-1].set_xticks([i for i in position1], models+ML_models+[''], rotation=35, fontsize=0.1)
    fig.legend(bplot['boxes'], models, loc=8, borderaxespad=2, ncol=4, shadow=False, fontsize=35)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig5/fig5_{cfg['case_name']}_{cfg['model_name']}_{cfg['testset']}_select.png", dpi=300)
    print(f"fig5: {cfg['case_name']} {cfg['model_name']} {cfg['testset']} Done -----")


def call_fun_by_str(cfg, y_lim, y):
    if cfg['selected'] == 'true':
        eval('box_plot_select')(cfg, y_lim, y)
    else:
        eval('box_plot')(cfg, y_lim, y)


if __name__ == "__main__":
    cfg = get_args()

    y_lim = [(0, 1), (0, 2), (0, 2), (-0.5, 1)]
    y = [0.5, 1,1, 0.5]
    call_fun_by_str(cfg, y_lim, y)

