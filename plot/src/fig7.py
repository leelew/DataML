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


def KGE_c(o, s):
    """
    Kling-Gupta Efficiency
    input:
        s: simulated
        o: observed
    output:
        kge: Kling-Gupta Efficiency
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    cc = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o)
    beta = np.sum(s) / np.sum(o)
    KGE = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return KGE  # , cc, alpha, beta


#
def train_data():
    stnlist = f"../spilt.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0

    r2 = np.full((3, len(station_list['i'])), np.nan)
    rmse = np.full((3, len(station_list['i'])), np.nan)
    kge = np.full((3, len(station_list['i'])), np.nan)
    models = ['DNN', 'LightGBM', 'Random Forest']
    Climate_zone = []

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        cz = xr.open_dataset('/tera07/zhwei/For_QingChen/DataML/FLUXNET/flux_all/%s.nc' % (station_list['filename'][i])).IGBP_veg_short
        Climate_zone.append(str(cz.values)[2:5])

        ds = xr.open_dataset('/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/train_more/train/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        r2[:, i] = np.array([r2_score(ET_sim, mdl_sim), r2_score(ET_sim, lgb_sim), r2_score(ET_sim, RF_sim)])
        rmse[:, i] = np.array(
            [np.sqrt(mean_squared_error(ET_sim, mdl_sim)), np.sqrt(mean_squared_error(ET_sim, lgb_sim)), np.sqrt(mean_squared_error(ET_sim, RF_sim))])
        kge[:, i] = np.array([KGE_c(ET_sim, mdl_sim), KGE_c(ET_sim, lgb_sim), KGE_c(ET_sim, RF_sim)])

    train_data = pd.DataFrame({'ID': np.tile(station_list['filename'], len(models) * 3),
                               'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                               'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                               'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                               'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    train_data.to_excel("/tera07/zhwei/For_QingChen/DataML/plot/fig7/train.xlsx", sheet_name='train', index=True)


def violin(train_data):
    fig, ax = plt.subplots(figsize=(18, 8))
    train_data.models.loc[~train_data['models'].isin(["LightGBM"])] = 'ET'
    train_data = train_data[train_data['Metric'].isin(["KGE"])]
    ax = sns.violinplot(x="Climate_zone", y="data", data=train_data, split=True, hue="models", scale='width', linewidth=1.5,
                        palette="Pastel1")  # hue="sex",

    ax.set_xlabel("Climate_zone", fontsize=16)
    ax.set_ylabel("KGE", fontsize=16)

    ax.set_title(f'train', fontsize=16, loc="left")
    ax.legend(fontsize=16, loc='best', shadow=False, frameon=False)

    plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig7/fig7_a.png", dpi=300)
    print('violinplot done')


# violin(train_data)


def box(train_data):
    labels = ["R2", "RMSE", "KGE"]
    models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']

    fig, axes = plt.subplots(figsize=(12, 6))
    position = np.arange(1, 2 * len(models), 2)
    print(position)

    colors = ['cadetblue', 'seagreen', 'lightcoral', '#F9E07F', '#7D58AD', '#898044']
    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        # print(model_data)
        R2_m = model_data[model_data['Metric'].isin(['R2'])]
        RMSE_m = model_data[model_data['Metric'].isin(['RMSE'])]
        KGE_m = model_data[model_data['Metric'].isin(['KGE'])]

        t_data = [R2_m.data.tolist(), RMSE_m.data.tolist(), KGE_m.data.tolist()]
        p = position[i]
        bplot = axes.boxplot(t_data, patch_artist=True, positions=(p, p + 0.5, p + 1), widths=0.3, labels=labels,
                             medianprops={'color': 'black', 'linewidth': '2.0'})
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # ax.yaxis.grid(True)
    axes.set_xticks([i + 0.5 for i in position], models)
    axes.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    axes.axhline(y=1, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
    axes.set(ylim=(0, 2))
    axes.legend(bplot['boxes'], labels, loc='best', shadow=False)
    axes.set_title(f'Train', fontsize=18, loc="left")

    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig7/fig7_case3_model5_a.png", dpi=300)
    plt.show()


def box_KGE(test_data, labels, y_lim):
    # labels = ["KGE"]
    # models = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    models = ['DNN', 'LightGBM', 'Random Forest']

    fig, axes = plt.subplots(figsize=(12, 6))
    position = np.arange(1, 2 * len(models), 2) + 1
    print(position)

    colors = ['cadetblue', 'seagreen', 'lightcoral', '#F9E07F', '#7D58AD', '#898044']
    KGE_data = train_data[train_data['Metric'].isin(labels)]
    DNN_m = KGE_data[KGE_data['models'].isin(['DNN'])]
    LightGBM_m = KGE_data[KGE_data['models'].isin(['LightGBM'])]
    # AutoML_m = KGE_data[KGE_data['models'].isin(['AutoML'])]
    RF_m = KGE_data[KGE_data['models'].isin(['Random Forest'])]

    t_data = [DNN_m.data.tolist(), LightGBM_m.data.tolist(), RF_m.data.tolist()]  # AutoML_m.data.tolist(),

    bplot = axes.boxplot(t_data, patch_artist=True, positions=position, widths=0.3,
                         medianprops={'color': 'black', 'linewidth': '2.0'})
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    for i, model in enumerate(models):
        df = KGE_data[KGE_data['models'].isin([model])].data
        axes.text(position[i] + 0.2, LightGBM_m.data.median() + 0.1, f"{df.median():.3f}", fontsize=20, c='black')

    axes.set_xticks([i for i in [1, 2, 4, 6, 7]], ['', 'DNN', 'LightGBM', 'Random Forest', ''])
    axes.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    axes.axhline(y=LightGBM_m.data.median(), ls="--", c="red", alpha=0.7)  # 添加水平直线 #105885
    axes.set(ylim=y_lim)
    axes.set_title(f'Train', fontsize=18, loc="left")

    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig7/fig7_case3_model6_{labels[0]}_a.png", dpi=300)
    plt.show()


def box_test(data, y_lim, y, cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    models = ['DNN', 'LightGBM','AutoML', 'Random Forest']
    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    position = np.arange(1, 2 * len(models), 2)
    colors = sns.color_palette("Set3", n_colors=len(Metric), desat=.7).as_hex()

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
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig7/fig7_{case_name}_{model_name}_{test_set}.png", dpi=300)
    print(f'fig7: {case_name} {model_name} {test_set} Done -----')


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
