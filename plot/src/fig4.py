import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import scipy
from scipy import stats
from statistics import mean
from pylab import rcParams
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
          'font.size': 15,
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

def compare(cfg):
    cfg = get_args()
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    data_site = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')
    data_time = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}_Timetest.xlsx", header=0, sheet_name=f'{test_set}_Timetest')

    data_site = data_site[data_site['models'].isin(['AutoML'])]
    data_site.loc[data_site["data"] > 1.5, "data"] = np.nan
    data_site.loc[data_site["data"] < -0.4, "data"] = np.nan
    data_site = data_site.dropna(subset=["data"])
    data_site['col'] = data_site['Metric']+ ' sitetest '

    data_time = data_time[data_time['models'].isin(['AutoML'])]
    data_time.loc[data_time["data"] > 1.5, "data"] = np.nan
    data_time.loc[data_time["data"] < -0.4, "data"] = np.nan
    data_time = data_time.dropna(subset=["data"])
    data_time['col'] =  data_time['Metric'] +' timetest '

    data = pd.concat((data_site, data_time), axis=0)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(4, rot=-.25, light=.7, start=2)  # 设置颜色渐变
    g = sns.FacetGrid(data, row="col", hue="col", aspect=15, height=.5, palette=pal)
    g.map(sns.kdeplot, "data", bw_adjust=.35, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "data", clip_on=False, color="w", lw=2, bw_adjust=.35)
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)


    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.05, .2, label, fontweight="bold", color='black',
                ha="left", va="center", transform=ax.transAxes, alpha=0.8)
        # ax.axhline(x=np.arange(-0.25, 1.5, 0.25), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        ax.grid(linestyle="--", alpha=0.5)  # 绘制图中虚线 透明度0.3


    g.map(label, "data")
    g.figure.subplots_adjust(hspace=-.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlabel="AutoML")
    g.despine(bottom=True, left=True)
    g.set(xlim=(-0.25, 1.5), ylim=(0, 3))
    # plt.tight_layout()
    plt.savefig(f"./fig4/fig4_{case_name}_{model_name}_{cfg['testset']}_compare.png", dpi=300)
    plt.show()


def kdeplot(cfg):
    cfg = get_args()
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    data = pd.read_excel(f"{cfg['excle_path']}/{case_name}_{model_name}_{test_set}.xlsx", header=0, sheet_name=f'{test_set}')

    data = data[data['models'].isin(['LightGBM', 'DNN', 'AutoML', 'Random Forest'])]
    data.loc[data["data"] > 1.5, "data"] = np.nan
    data.loc[data["data"] < -0.4, "data"] = np.nan
    data = data.dropna(subset=["data"])
    data['col'] = data['models'] + ' ' + data['Metric']

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(4, rot=-.25, light=.7, start=2)  # 设置颜色渐变
    g = sns.FacetGrid(data, row="col", hue="col", aspect=15, height=.5, palette=pal)
    g.map(sns.kdeplot, "data", bw_adjust=.35, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "data", clip_on=False, color="w", lw=2, bw_adjust=.35)
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)


    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.05, .2, label, fontweight="bold", color='black',
                ha="left", va="center", transform=ax.transAxes, alpha=0.8)
        # ax.axhline(x=np.arange(-0.25, 1.5, 0.25), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        ax.grid(linestyle="--", alpha=0.5)  # 绘制图中虚线 透明度0.3


    g.map(label, "data")
    g.figure.subplots_adjust(hspace=-.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlabel="Train")
    g.despine(bottom=True, left=True)
    g.set(xlim=(-0.25, 1.5), ylim=(0, 3))
    # plt.tight_layout()
    plt.savefig(f"./fig4/fig4_{case_name}_{model_name}_{cfg['testset']}.png", dpi=300)
    plt.show()

def call_fun_by_str(cfg):
    if cfg['selected'] == 'compare':
        eval('compare')(cfg)
    else:
        eval('kdeplot')(cfg)

if __name__ == "__main__":
    cfg = get_args()
    call_fun_by_str(cfg)
    # kdeplot(cfg)
    # compare(cfg)
