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

### Plot settings
font = {'family': 'Times new roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 18,
          'grid.linewidth': 0.2,
          'font.size': 18,
          'legend.fontsize': 16,
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



if __name__ == "__main__":
    # train_data()
    # test_data()

    stnlist = f"../case3_M.xlsx"
    train = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0
    test = pd.read_excel(stnlist, header=0, sheet_name='test')
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
              'PMLV2', 'DNN', 'LightGBM', 'Random Forest']
    # models = ['DNN', 'LightGBM','AutoML', 'Random Forest']

    stnlist1 = f"../case3.xlsx"
    station_train = pd.read_excel(stnlist1, header=0, sheet_name='train')  # ,header=0
    station_test = pd.read_excel(stnlist1, header=0, sheet_name='test')  # ,header=0

    labellist_train, labellist_test = [], []
    for i in np.arange(len(station_train['filename'])):
        labellist_train.append(station_train['filename'][i][:6])

    for i in np.arange(len(station_test['filename'])):
        labellist_test.append(station_test['filename'][i][:6])

    train_r2, test_r2 = [], []
    train_RMSE, test_RMSE = [], []
    train_KGE, test_KGE = [], []
    for model in models:
        train_r2.append(train[train['models'].isin([model])].R2.tolist())
        test_r2.append(test[test['models'].isin([model])].R2.tolist())
        train_RMSE.append(train[train['models'].isin([model])].RMSE.tolist())
        test_RMSE.append(test[test['models'].isin([model])].RMSE.tolist())
        train_KGE.append(train[train['models'].isin([model])].KGE.tolist())
        test_KGE.append(test[test['models'].isin([model])].KGE.tolist())

    train_r2, test_r2 = np.array(train_r2), np.array(test_r2)
    train_RMSE, test_RMSE = np.array(train_RMSE), np.array(test_RMSE)
    train_KGE, test_KGE = np.array(train_KGE), np.array(test_KGE)

    # train_r2 = np.where((train_r2 < 2) , train_r2, 2.1)
    # test_r2 = np.where((test_r2 > -1), test_r2, -1.1)
    # train_r2 = np.where((train_r2 < 2) , train_r2, 2.1)
    # test_r2 = np.where((test_r2 > -1), test_r2, -1.1)
    # train_RMSE = np.where((train_RMSE < 10) & (train_RMSE > -10), train_RMSE, 0)
    # test_RMSE = np.where((test_RMSE < 10) & (test_RMSE > -10), test_RMSE, 0)
    # train_KGE = np.where((train_KGE < 10) & (train_KGE > -10), train_KGE, 0)
    # test_KGE = np.where((test_KGE < 10) & (test_KGE > -10), test_KGE, 0)

    fig, ax1 = plt.subplots(figsize=(73, 16))
    ax1 = sns.heatmap(data=train_r2, vmin=0, vmax=1, linewidth=.5, cmap=sns.diverging_palette(20, 220, n=200), center=0.5,
                      xticklabels=labellist_train,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax1.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax1.set_xlabel('In situ', fontsize=16)
    ax1.set_title(f'R2', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_train_R2.png", dpi=300)

    fig, ax2 = plt.subplots(figsize=(43, 16))
    ax2 = sns.heatmap(data=test_r2, vmin=0, vmax=1, linewidth=.5, cmap=sns.diverging_palette(20, 220, n=200), center=0.5, xticklabels=labellist_test,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax2.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax2.set_xlabel('In situ', fontsize=16)
    ax2.set_title(f'R2', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_test_R2.png", dpi=300)

    fig, ax3 = plt.subplots(figsize=(73, 16))
    ax3 = sns.heatmap(data=train_RMSE, vmin=0, vmax=2, linewidth=.5, cmap=sns.diverging_palette(220, 20, n=200), center=1,
                      xticklabels=labellist_train,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax3.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax3.set_xlabel('In situ', fontsize=16)
    ax3.set_title(f'RMSE', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_train_RMSE.png", dpi=300)

    fig, ax4 = plt.subplots(figsize=(43, 16))
    ax4 = sns.heatmap(data=test_RMSE, vmin=0, vmax=2, linewidth=.5, cmap=sns.diverging_palette(220, 20, n=200), center=1, xticklabels=labellist_test,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax4.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax4.set_xlabel('In situ', fontsize=16)
    ax4.set_title(f'RMSE', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_test_RMSE.png", dpi=300)

    fig, ax5 = plt.subplots(figsize=(73, 16))
    ax5 = sns.heatmap(data=train_KGE, vmin=-1, vmax=1, linewidth=.5, cmap=sns.diverging_palette(20, 220, n=200), center=0,
                      xticklabels=labellist_train,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax5.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax5.set_xlabel('In situ', fontsize=16)
    ax5.set_title(f'KGE', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_train_KGE.png", dpi=300)

    fig, ax6 = plt.subplots(figsize=(43, 16))
    ax6 = sns.heatmap(data=test_KGE, vmin=-1, vmax=1, linewidth=.5, cmap=sns.diverging_palette(20, 220, n=200), center=0,
                      xticklabels=labellist_test,
                      yticklabels=models,
                      cbar_kws={"pad": 0.02})
    ax6.set_ylabel('ET product', fontsize=16)  # x轴标题
    ax6.set_xlabel('In situ', fontsize=16)
    ax6.set_title(f'KGE', fontsize=18, loc="left")

    plt.tight_layout()
    plt.savefig(f"./case3_test_KGE.png", dpi=300)
