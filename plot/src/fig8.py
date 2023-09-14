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
          'font.size': 18,
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


def box_plot_time(data, y_lim, y, cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    data_types = ['BSV', 'CRO', 'CSH', 'CVM', 'DBF', 'DNF', 'EBF', 'ENF', 'GRA', 'MF', 'OSH', 'SAV', 'WET', 'WSA']
    position = np.arange(1, 2 * len(data_types), 2)
    colors = sns.color_palette("Set3", n_colors=len(data_types), desat=.7).as_hex()

    for k, mm in enumerate(Metric):
        Metric_data = data[data['Metric'].isin([mm])]
        fig, axes = plt.subplots(4, figsize=(16, 15), sharex=True)
        for i, model in enumerate(models):
            model_data = Metric_data[Metric_data['models'].isin([model])]
            t_data = []
            for t, data_type in enumerate(data_types):
                # try
                t_data.append(model_data[model_data['Climate_zone'].isin([data_type])].data)

            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=position,
                                    widths=0.4, medianprops={'color': 'black', 'linewidth': '2.0'},
                                    capprops={"color": "black", "linewidth": 2.0},
                                    whiskerprops={"color": "black", "linewidth": 2.0})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            for t, data_type in enumerate(data_types):
                df = model_data[model_data['Climate_zone'].isin([data_type])].data
                if df.median() < -1:
                    axes[i].text(position[t] + 0.31, y[k] + 0.1, f"", fontsize=20, c='black')
                else:
                    axes[i].text(position[t] + 0.31, y[k] + 0.1, f"{df.median():.3f}", fontsize=20, c='black')

            axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
            axes[i].axhline(y=y[k], ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
            axes[i].set(ylim=y_lim[k])
            axes[i].set_ylabel(f'{model}', fontsize=20, )
            axes[0].set_title(f'{cfg["tittle"]}', fontsize=20, loc="left")
            position1 = np.append(position, values=position[-1] + 2)
            axes[i].set_xticks([m for m in position1], data_types+[' '], fontsize=20)
        # fig.legend(bplot['boxes'], models, loc=8, borderaxespad=2, ncol=4, shadow=False, fontsize=35)
        # plt.subplots_adjust(hspace=0.1)
        plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig8/fig8_{case_name}_{model_name}_{test_set}_{mm}.png", dpi=300)
        print(f'fig8: {case_name} {model_name} {test_set} {mm} Done -----')


def box_plot(data, y_lim, y, cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    data_types = ['BSV', 'CRO', 'CSH', 'CVM', 'DBF', 'DNF', 'EBF', 'ENF', 'GRA', 'MF', 'OSH', 'SAV', 'WET', 'WSA']
    position = np.arange(1, 2 * len(data_types), 2)
    colors = sns.color_palette("Set3", n_colors=len(data_types), desat=.7).as_hex()

    for k, mm in enumerate(Metric):
        Metric_data = data[data['Metric'].isin([mm])]
        fig, axes = plt.subplots(4, figsize=(16, 15))
        for i, model in enumerate(models):
            model_data = Metric_data[Metric_data['models'].isin([model])]
            t_data = []
            for t, data_type in enumerate(data_types):
                # try
                t_data.append(model_data[model_data['Climate_zone'].isin([data_type])].data)

            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=position,
                                    widths=0.4, medianprops={'color': 'black', 'linewidth': '2.0'},
                                    capprops={"color": "black", "linewidth": 2.0},
                                    whiskerprops={"color": "black", "linewidth": 2.0})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            for  t, data_type in enumerate(data_types):
                df = model_data[model_data['Climate_zone'].isin([data_type])].data
                if df.median() < -1:
                    axes[i].text(position[t] + 0.31, y[k] + 0.1, f"", fontsize=20, c='black')
                else:
                    axes[i].text(position[t] + 0.31, y[k] + 0.1, f"{df.median():.3f}", fontsize=20, c='black')


            axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
            axes[i].axhline(y=y[k], ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
            axes[i].set(ylim=y_lim[k])
            axes[i].set_ylabel(f'{model}', fontsize=20, )
            axes[0].set_title(f'{cfg["tittle"]}', fontsize=20, loc="left")
            position1 = np.append(position, values=position[-1] + 2)
            # print(position,position1)
            axes[i].set_xticks([m for m in position1], data_types+[' '], fontsize=20)
            # fig.legend(bplot['boxes'], models, loc=8, borderaxespad=2, ncol=4, shadow=False, fontsize=35)
            # plt.subplots_adjust(hspace=0.1)
        plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig8/fig8_{case_name}_{model_name}_{test_set}_{mm}.png", dpi=300)
        print(f'fig8: {case_name} {model_name} {test_set} {mm} Done -----')


if __name__ == "__main__":
    cfg = get_args()
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    input_data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')

    y_lim = [(0, 1), (0, 2), (0, 2), (-0.5, 1)]
    y = [0.5, 1, 1, 0.5]
    bottom = [0, 0, 0, -0.5]
    if cfg['test_case'] == 'time test':
        eval('box_plot_time')(input_data, y_lim, y, cfg)
    else:
        eval('box_plot')(input_data, y_lim, y, cfg)
