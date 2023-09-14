import os

import scipy
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from pylab import rcParams
from statistics import mean
from sklearn.metrics import r2_score, mean_squared_error
import argparse
from config import get_args
from pathlib import PosixPath, Path

### Plot settings
font = {'family': 'Times New Roman'}
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


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = (time_end - time_start) / 3600.
        print('%s cost time: %.3f hours' % (func.__name__, time_spend))
        return result

    return func_wrapper


def corrlation_KGE(o, s):
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


def corrlation_R2(o, s):
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


def scatter_plot(data, case_name, model_name, run, run_name, cfg):
    n = len(data['filename'])
    test_set = cfg['testset']
    ET_sim = []
    lgb_sim = []
    for i, m in enumerate(data['filename']):
        filevar = f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/{test_set}/{m}.nc'
        if os.path.exists(filevar):
            ds = xr.open_dataset(filevar)
            ET_sim.append(ds.et.values)
            lgb_sim.append(ds[f'{run}'].values)

    ET_sim = np.concatenate(ET_sim, axis=0)
    lgb_sim = np.concatenate(lgb_sim, axis=0)
    lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
    ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
    lgb_sim = np.delete(lgb_sim, np.argwhere(ET_sim < -1), axis=0)
    ET_sim = np.delete(ET_sim, np.argwhere(ET_sim < -1), axis=0)

    r2 = corrlation_R2(ET_sim, lgb_sim)
    mse = mean_squared_error(ET_sim, lgb_sim) # r = np.corrcoef(ET_sim, lgb_sim)[0, 1]
    rmse = np.sqrt(mean_squared_error(ET_sim, lgb_sim))
    KGE = corrlation_KGE(ET_sim, lgb_sim)

    xy = np.vstack([ET_sim, lgb_sim])
    z = stats.gaussian_kde(xy)(xy)
    # ===========Sort the points by density, so that the densest points are plotted last===========
    idx = z.argsort()
    x, y, z = ET_sim[idx], lgb_sim[idx], z[idx]

    def best_fit_slope_and_intercept(xs, ys):
        m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
        b = np.mean(ys) - m * np.mean(xs)
        return m, b

    m, b = best_fit_slope_and_intercept(x, y)
    regression_line = []
    for a in range(0, 10):
        regression_line.append((m * a) + b)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap4 = mpl.colormaps.get_cmap('Spectral_r')
    norm4 = mcolors.TwoSlopeNorm(vmin=0, vmax=40, vcenter=20)

    scatter = ax.scatter(x, y, marker='o', c=z * 100, s=1, cmap=cmap4, norm=norm4, alpha=0.8)  # cm.batlow

    plt.plot(np.arange(0, 10), regression_line, 'red', lw=1.5)  # 预测与实测数据之间的回归线
    cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', aspect=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Density', fontdict={'size': 17})  # 设置colorbar的标签字体及其大小

    ax.plot([-1, 9], [-1, 9], c="black", ls='--')
    ax.set_yticks(np.arange(0, 10, 2))
    ax.set_yticklabels(np.arange(0, 10, 2), fontsize=19)
    ax.set_xticks(np.arange(0, 10, 2))
    ax.set_xticklabels(np.arange(0, 10, 2), fontsize=19)
    ax.set_xlabel("Observed ET [mm/day]", fontsize=20)
    ax.set_ylabel("Simulated ET [mm/day]", fontsize=20)

    ax.set_title(f'{cfg["tittle"]} {run_name}', fontsize=18, loc="left")
    ax.text(.1, .95, f"N={len(ET_sim)}", transform=ax.transAxes, fontsize=20, zorder=4)
    ax.text(.1, .91, f"R2={r2:.3f}", transform=ax.transAxes, fontsize=20, zorder=4)
    ax.text(.1, .87, f"MSE={mse:.3f}", transform=ax.transAxes, fontsize=20, zorder=4)
    ax.text(.1, .83, f"KGE={KGE:.3f}", transform=ax.transAxes, fontsize=20, zorder=4)
    ax.text(6, 1, f"Y = {m:.3f}X + {b:.3f}", fontsize=20, va='bottom', c='black')
    ax.set(xlim=(0, 9), ylim=(-0.3, 9))

    plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig3/fig3_{case_name}_{model_name}_{run_name}_{test_set}.png", dpi=300)
    # plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig3/fig3_{case_name}_{model_name}_{run_name}_{test_set}.eps", dpi=300)
    print(f"fig3: {case_name} {model_name} {run_name} {test_set} Done ---------")


if __name__ == "__main__":
    # load data
    cfg = get_args()
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    if cfg['test_case'] == 'site test':
        data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}.xlsx", header=0, sheet_name=f'{test_set}')

    if cfg['test_case'] == 'time test':
        data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}.xlsx", header=0, sheet_name='time_test')

    # run = 'gbm'
    # run_name = 'LightGBM'
    # scatter_plot(data, case_name, model_name, run, run_name, cfg)
    #
    # run = 'mdl'
    # run_name = 'DNN'
    # scatter_plot(data, case_name, model_name, run, run_name, cfg)
    #
    # run = 'RF'
    # run_name = 'Random_Forest'
    # scatter_plot(data, case_name, model_name, run, run_name, cfg)

    run = 'am'
    run_name = 'AutoML'
    scatter_plot(data, case_name, model_name, run, run_name, cfg)
