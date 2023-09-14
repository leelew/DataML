import math
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import glob, os, shutil, sys
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
          'axes.labelsize': 18,
          'grid.linewidth': 0.2,
          'font.size': 16,
          'legend.fontsize': 18,
          'legend.frameon': False,
          'xtick.labelsize': 18,
          'xtick.direction': 'out',
          'ytick.labelsize': 18,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def box_test(data, y_lim, y, cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    # models = ['DNN', 'LightGBM','AutoML', 'Random Forest']
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b',  'REA', 'ERA5', 'GLEAM_hybrid','FLUXCOM_9km','DNN', 'LightGBM','AutoML', 'Random Forest']
    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    position = np.arange(1, 2 * len(models), 2)
    colors = sns.color_palette("Set3", n_colors=len(models), desat=.7).as_hex()

    for i, mm in enumerate(Metric):
        Metric_data = data[data['Metric'].isin([mm])]
        train_data = []
        for model in (models):
            train_data.append(Metric_data[Metric_data['models'].isin([model])].data)

        bplot = axes[i].boxplot(train_data, patch_artist=True, positions=position,
                                 widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                 capprops={"color": "black", "linewidth": 2.0},
                                 whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for j, model in zip(np.arange(0, len(models)), models):
            df = Metric_data[Metric_data['models'].isin([model])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')

        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=Metric_data[Metric_data['models'].isin(['AutoML'])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=y_lim[i])
        axes[i].set_ylabel(f'{mm}', fontsize=40, )

    position1 = np.append(position, values=position[-1] + 1)
    axes[-1].set_xticks([i for i in position1], models+[''], rotation=35, fontsize=0.1)
    fig.legend(bplot['boxes'], models, loc=8, borderaxespad=2, ncol=4, shadow=False, fontsize=35)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case_name}_{model_name}_{test_set}.png", dpi=300)
    print(f'fig6: {case_name} {model_name} {test_set} Done -----')


if __name__ == "__main__":
    cfg = get_args()
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    input_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')


    input_data = input_data.dropna(subset=["data"])
    # box(train_data)
    # box_KGE(train_data)

    y_lim = [(0, 1), (0, 2), (0, 2), (-0.5, 1)]
    y = [0.5, 1, 1, 0.5]
    bottom = [0, 0, 0, -0.5]

    box_test(input_data,y_lim, y, cfg)

    # labels = ["KGE"]
    # box_KGE(train_data, labels, (-0.4, 1))
    #
    # labels = ["R2"]
    # box_KGE(train_data, labels, (0, 1))
    #
    # labels = ["RMSE"]
    # box_KGE(train_data, labels, (0, 2))
