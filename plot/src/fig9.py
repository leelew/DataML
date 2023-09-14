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


def data_prepare(model, compare_lists, name_lists, inform_lists):
    data = []
    for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
        case = f"../xlsx/{name_list}_{compare_list}_train.xlsx"
        case_data = pd.read_excel(case, header=0, sheet_name='train')  # ,header=0
        case_ml = case_data[case_data['models'].isin(model)]  # LightGBM/ AutoML
        case_ml['col'] = case_ml['models'] + ' ' + f'Train {name_list}_{compare_list}:\n{inform}'
        case_ml = case_ml.dropna(subset=["data"])
        data.append(case_ml)
    train_data = pd.concat((case_ml for case_ml in data), axis=0)

    data = []
    for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
        case = f"../xlsx/{name_list}_{compare_list}_test.xlsx"
        case_data = pd.read_excel(case, header=0, sheet_name='test')  # ,header=0
        case_ml = case_data[case_data['models'].isin(model)]  # LightGBM/ AutoML
        case_ml['col'] = case_ml['models'] + ' ' + f'Test {name_list}_{compare_list}:\n{inform}'
        case_ml = case_ml.dropna(subset=["data"])
        data.append(case_ml)
    test_data = pd.concat((case_ml for case_ml in data), axis=0)

    outdata = pd.concat((train_data, test_data), axis=0)
    return outdata


def box(model, compare_lists, name_lists, inform_lists, y_lim, y):
    outdata = data_prepare(model, compare_lists, name_lists, inform_lists)
    print(outdata)
    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    position = np.arange(1, 4 * len(compare_lists), 2)
    colors = sns.color_palette("Set3", n_colors=len(compare_lists) * 2, desat=.7).as_hex()

    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    for i, mm in enumerate(Metric):
        xlabel,label, train_label, test_label =[], [], [], []
        Metric_data = outdata[outdata['Metric'].isin([mm])]
        t_data = []
        for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Train {name_list}_{compare_list}:\n{inform}'
            label.append(f'Train {name_list}_{compare_list}:\n{inform}'), train_label.append(sel)
            xlabel.append(f'{name_list}_{compare_list}')
            t_data.append(Metric_data[Metric_data['col'].isin([sel])].data)
        for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Test {name_list}_{compare_list}:\n{inform}'
            t_data.append(Metric_data[Metric_data['col'].isin([sel])].data)
            label.append(f'Test {name_list}_{compare_list}:\n{inform}'), test_label.append(sel)
            xlabel.append(f'{name_list}_{compare_list}')
        xlabel.append('')
        bplot = axes[i].boxplot(t_data, patch_artist=True, positions=position,
                                widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                capprops={"color": "black", "linewidth": 2.0},
                                whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for j, name_list, compare_list, inform in zip(np.arange(0, len(compare_lists)), name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Train {name_list}_{compare_list}:\n{inform}'
            df = Metric_data[Metric_data['col'].isin([sel])].data
            axes[i].text(position[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')
            k = j + len(compare_lists)
            sel = f"{model[0]}" + ' ' + f'Test {name_list}_{compare_list}:\n{inform}'
            df = Metric_data[Metric_data['col'].isin([sel])].data
            axes[i].text(position[k] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')

        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=Metric_data[Metric_data['col'].isin([train_label[-1]])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].axhline(y=Metric_data[Metric_data['col'].isin([test_label[-1]])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].axvline(x=position[len(compare_lists)]-1, ls="--", c="black", alpha=0.7)
        axes[i].set(ylim=y_lim[i])
        axes[i].set_ylabel(f'{mm}', fontsize=40, )

    position1 = np.append(position, values=position[-1] + 2)
    axes[-1].set_xticks([i for i in position1], xlabel, rotation=35,fontsize=0.1)
    # axes[1].legend(bplot['boxes'], label,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, shadow=False, fontsize=30, title_fontsize=35)
    fig.legend(bplot['boxes'], label,  loc=8, borderaxespad=2, ncol=len(compare_lists), shadow=False, fontsize=25)
    plt.subplots_adjust(hspace=0.1)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig9/fig9_{model[0]}.png", dpi=300)  # LightGBM/ AutoML
    print('boxplot done')

def box_test(model, compare_lists, name_lists, inform_lists, y_lim, y, bottom,end):
    outdata = data_prepare(model, compare_lists, name_lists, inform_lists)
    print(outdata)
    fig, axes = plt.subplots(4, figsize=(35, 38), sharex=True)
    xposition = np.arange(1, 4 * len(compare_lists), 2)
    position = np.arange(1, 2 * len(compare_lists), 2)
    colors = sns.color_palette("Set3", n_colors=len(compare_lists), desat=.7).as_hex()

    Metric = ['R2', 'MSE', 'RMSE', 'KGE']
    for i, mm in enumerate(Metric):
        xlabel,label, train_label, test_label =[], [], [], []
        Metric_data = outdata[outdata['Metric'].isin([mm])]
        train_data,test_data = [],[]
        for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Train {name_list}_{compare_list}:\n{inform}'
            label.append(f'{name_list}_{compare_list}:\n{inform}'), train_label.append(sel)
            xlabel.append(f'{name_list}_{compare_list}')
            train_data.append(Metric_data[Metric_data['col'].isin([sel])].data)
        for name_list, compare_list, inform in zip(name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Test {name_list}_{compare_list}:\n{inform}'
            xlabel.append(f'{name_list}_{compare_list}'),test_label.append(sel)
            test_data.append(Metric_data[Metric_data['col'].isin([sel])].data)
        xlabel.append('')
        bplot1 = axes[i].boxplot(train_data, patch_artist=True, positions=xposition[:len(compare_lists)],
                                widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                capprops={"color": "black", "linewidth": 2.0},
                                whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)
        bplot2 = axes[i].boxplot(test_data, patch_artist=True, positions=xposition[len(compare_lists):],
                                widths=0.6, medianprops={'color': 'black', 'linewidth': '2.0'},
                                capprops={"color": "black", "linewidth": 2.0},
                                whiskerprops={"color": "black", "linewidth": 2.0})

        for patch, color in zip(bplot2['boxes'], colors):
            patch.set_facecolor(color)

        for j, name_list, compare_list, inform in zip(np.arange(0, len(compare_lists)), name_lists, compare_lists, inform_lists):
            sel = f"{model[0]}" + ' ' + f'Train {name_list}_{compare_list}:\n{inform}'
            df = Metric_data[Metric_data['col'].isin([sel])].data
            axes[i].text(xposition[j] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')
            k = j + len(compare_lists)
            sel = f"{model[0]}" + ' ' + f'Test {name_list}_{compare_list}:\n{inform}'
            df = Metric_data[Metric_data['col'].isin([sel])].data
            axes[i].text(xposition[k] + 0.31, y[i] + 0.1, f"{df.median():.3f}", fontsize=30, c='black')

        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=Metric_data[Metric_data['col'].isin([train_label[-1]])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].axhline(y=Metric_data[Metric_data['col'].isin([test_label[-1]])].data.median(), ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].axvline(x=xposition[len(compare_lists)]-1, ls="--", c="black", alpha=0.7)
        axes[i].set(ylim=y_lim[i])
        axes[i].set_ylabel(f'{mm}', fontsize=40, )
        axes[i].text(xposition[0], bottom[i]+0.05, f"Train",  fontsize=26, zorder=4)
        axes[i].text(xposition[len(compare_lists)],bottom[i]+0.05 , f"Test",  fontsize=26, zorder=4)

    position1 = np.append(xposition, values=xposition[-1] + 2)
    axes[-1].set_xticks([i for i in position1], xlabel, rotation=35,fontsize=0.1)
    # axes[1].legend(bplot['boxes'], label,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, shadow=False, fontsize=30, title_fontsize=35)
    n = len(compare_lists)
    if n >=6:
        n = int(n/2)+1
    fig.legend(bplot1['boxes'], label,  loc=8, borderaxespad=2, ncol=n, shadow=False, fontsize=30)
    plt.subplots_adjust(hspace=0.1)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig9/fig9_{model[0]}_{end}.png", dpi=300)  # LightGBM/ AutoML
    print('boxplot done')

if __name__ == "__main__":
    model = ['Random Forest']
    y_lim = [(0, 1), (0, 2), (0, 2), (-0.5, 1)]
    y = [0.5, 1, 1, 0.5]
    bottom = [0, 0, 0, -0.5]

    compare_list = ['model3', 'model8', 'model9', 'model1','model2','model3','model4']
    name_list = ['case3', 'case3', 'case3', 'case5','case5','case5','case5']
    inform_lists = ['FLAML', 'FLAML fit with rmse', 'FLAML: holdout','H2o Training', 'FLAML Training',
                    'FLAML: one ET product','FLAML: only ET products']
    box_test(model, compare_list, name_list, inform_lists, y_lim, y,bottom,'all')

    # compare_list = ['model3', 'model8', 'model9']
    # name_list = ['case3', 'case3', 'case3']
    # inform_lists = ['FLAML', 'FLAML fit with rmse', 'FLAML: holdout']
    # box_test(model, compare_list, name_list, inform_lists, y_lim, y,bottom,'case3')
    #
    #
    # compare_list = ['model9', 'model2','model3','model4']
    # name_list = ['case3', 'case5','case5','case5']
    # inform_lists = [ 'FLAML: holdout', 'FLAML Training','FLAML: one ET product','FLAML: only ET products']
    # box_test(model, compare_list, name_list, inform_lists, y_lim, y,bottom,'FLAML_holdout')
    #
    # compare_list = [ 'model1','model2']
    # name_list = ['case5','case5']
    # inform_lists = ['H2o Training', 'FLAML Training']
    # box_test(model, compare_list, name_list, inform_lists, y_lim, y,bottom,'compare')

    # compare_list = ['model2','model3','model4']
    # name_list = ['case5','case5','case5']
    # inform_lists = ['FLAML Training','FLAML: one ET product','FLAML: only ET products']
    # box_test(model, compare_list, name_list, inform_lists, y_lim, y,bottom,'change_input')