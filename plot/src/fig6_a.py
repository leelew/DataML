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


def train_data():
    stnlist = f"../spilt.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0

    r2 = np.full((12, len(station_list['i'])), np.nan)
    rmse = np.full((12, len(station_list['i'])), np.nan)
    kge = np.full((12, len(station_list['i'])), np.nan)
    Climate_zone = []
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
              'PMLV2']
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/flux/' + f"{station_list['filename'][i]}.nc"
        ET = np.array(xr.open_dataset(indata).LE.squeeze("x").squeeze("y"))
        cz = xr.open_dataset('/tera07/zhwei/For_QingChen/DataML/FLUXNET/flux_all/%s.nc' % (station_list['filename'][i])).IGBP_veg_short
        os.chdir("/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/ET/")
        for j, model in enumerate(models):
            filename = glob.glob(f"./{model}/{station_list['filename'][i][:6]}*.nc", recursive=True)
            # Et = xr.open_dataset(filename[0]).ET
            data = np.array(xr.open_dataset(filename[0]).ET)
            obs = np.array(ET)
            if np.count_nonzero(~np.isnan(data)) < len(data):
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(data)), axis=0)
                data = np.delete(np.array(data), np.argwhere(np.isnan(data)), axis=0)
            if np.count_nonzero(~np.isnan(obs)) < len(obs):
                data = np.delete(np.array(data), np.argwhere(np.isnan(obs)), axis=0)
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(obs)), axis=0)
            try:
                r2[j, i] = r2_score(obs, data)
                rmse[j, i] = np.sqrt(mean_squared_error(obs, data))
                kge[j, i] = KGE_c(obs, data)
            except ValueError as err:
                if str(err) == 'Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is require':
                    r2[j, i] = 0.
                    rmse[j, i] = 0.
                    kge[j, i] = 0.

        ds = xr.open_dataset('/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/train_more/train/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim = ds.et.values, ds.gbm.values
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        r2[11, i] = r2_score(ET_sim, lgb_sim)
        rmse[11, i] = np.sqrt(mean_squared_error(ET_sim, lgb_sim))
        kge[11, i] = KGE_c(ET_sim, lgb_sim)
        Climate_zone.append(str(cz.values)[2:5])

    models.append('LightGBM')
    train_data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 3),
                               'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                               'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                               'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                               'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    train_data.to_excel("/tera07/zhwei/For_QingChen/DataML/plot/fig6/train.xlsx", sheet_name='train', index=True)


def violin(train_data):
    fig, ax = plt.subplots(figsize=(18, 8))
    train_data.models.loc[~train_data['models'].isin(["LightGBM"])] = 'ET'
    train_data = train_data[train_data['Metric'].isin(["KGE"])]
    ax = sns.violinplot(x="Climate_zone", y="data", data=train_data, split=True, hue="models", scale='width', linewidth=1.5,
                        palette="Pastel1")  # hue="sex",

    ax.set_xlabel("Climate_zone", fontsize=16)
    ax.set_ylabel("KGE", fontsize=16)

    ax.set_title(f'Train', fontsize=16, loc="left")
    ax.legend(fontsize=16, loc='best', shadow=False, frameon=False)

    plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_a.eps", dpi=300)
    print('violinplot done')


def box(train_data, case_name, model_name):
    labels = ["R2", "RMSE", "KGE"]
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid']
    # data_types = ['SAV', 'WSA', 'GRA', 'EBF', 'MF ', 'ENF', 'OSH', 'CRO', 'WET', 'DBF', 'CSH']
    data_types = ['SAV', 'WSA', 'GRA', 'EBF', 'MF ', 'ENF', 'OSH', 'CRO', 'WET', 'DBF', 'CSH', 'DNF', 'BSV']

    position = np.arange(1, 3 * len(data_types), 3)
    # print(position)
    colors = ['cadetblue', 'seagreen', 'lightcoral', '#F9E07F', '#7D58AD', '#898044']
    fig, axes = plt.subplots(3, figsize=(16, 15))
    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        R2_m = model_data[model_data['Metric'].isin(['R2'])]
        RMSE_m = model_data[model_data['Metric'].isin(['RMSE'])]
        KGE_m = model_data[model_data['Metric'].isin(['KGE'])]

        for t, data_type in enumerate(data_types):
            t_data = [R2_m[R2_m['Climate_zone'].isin([data_type])].data.tolist(), RMSE_m[RMSE_m['Climate_zone'].isin([data_type])].data.tolist(),
                      KGE_m[KGE_m['Climate_zone'].isin([data_type])].data.tolist()]
            p = position[t]
            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=(p, p + 1, p + 2),
                                    widths=0.5, labels=labels, medianprops={'color': 'black', 'linewidth': '2.0'})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        axes[i].set_xticks([i + 0.5 for i in position], data_types)
        axes[i].axhline(y=-0.4, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=(-0.5, 2))
        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=1, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set_ylabel(f'{model}', fontsize=18)
    axes[0].set_title(f'Train', fontsize=30, loc="left")
    # fig.legend(bplot['boxes'], labels, loc='lower center', ncol=3)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case_name}_{model_name}_a1.eps", dpi=300)

    models = ['REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1']
    fig, axes = plt.subplots(3, figsize=(16, 15))
    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        R2_m = model_data[model_data['Metric'].isin(['R2'])]
        RMSE_m = model_data[model_data['Metric'].isin(['RMSE'])]
        KGE_m = model_data[model_data['Metric'].isin(['KGE'])]

        for t, data_type in enumerate(data_types):
            t_data = [R2_m[R2_m['Climate_zone'].isin([data_type])].data.tolist(), RMSE_m[RMSE_m['Climate_zone'].isin([data_type])].data.tolist(),
                      KGE_m[KGE_m['Climate_zone'].isin([data_type])].data.tolist()]
            p = position[t]
            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=(p, p + 1, p + 2),
                                    widths=0.5, labels=labels, medianprops={'color': 'black', 'linewidth': '2.0'})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        axes[i].set_xticks([i + 0.5 for i in position], data_types)
        axes[i].axhline(y=-0.4, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=(-0.5, 2))
        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=1, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set_ylabel(f'{model}', fontsize=18)
    axes[0].set_title(f'Train', fontsize=30, loc="left")
    # fig.legend(bplot['boxes'], labels, loc='lower center', ncol=3)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case_name}_{model_name}_a2.eps", dpi=300)

    models = ['ERA5', 'ET_3T', 'EB_ET']
    fig, axes = plt.subplots(3, figsize=(16, 15))
    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        R2_m = model_data[model_data['Metric'].isin(['R2'])]
        RMSE_m = model_data[model_data['Metric'].isin(['RMSE'])]
        KGE_m = model_data[model_data['Metric'].isin(['KGE'])]

        for t, data_type in enumerate(data_types):
            t_data = [R2_m[R2_m['Climate_zone'].isin([data_type])].data.tolist(), RMSE_m[RMSE_m['Climate_zone'].isin([data_type])].data.tolist(),
                      KGE_m[KGE_m['Climate_zone'].isin([data_type])].data.tolist()]
            p = position[t]
            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=(p, p + 1, p + 2),
                                    widths=0.5, labels=labels, medianprops={'color': 'black', 'linewidth': '2.0'})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        axes[i].set_xticks([i + 0.5 for i in position], data_types)
        axes[i].axhline(y=-0.4, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=(-0.5, 2))
        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=1, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set_ylabel(f'{model}', fontsize=18)
    axes[0].set_title(f'Train', fontsize=30, loc="left")
    # fig.legend(bplot['boxes'], labels, loc='lower center', ncol=3)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case_name}_{model_name}_a3.eps", dpi=300)

    models = ['FLUXCOM_9km', 'PMLV2']
    fig, axes = plt.subplots(3, figsize=(16, 15))
    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        R2_m = model_data[model_data['Metric'].isin(['R2'])]
        RMSE_m = model_data[model_data['Metric'].isin(['RMSE'])]
        KGE_m = model_data[model_data['Metric'].isin(['KGE'])]

        for t, data_type in enumerate(data_types):
            t_data = [R2_m[R2_m['Climate_zone'].isin([data_type])].data.tolist(), RMSE_m[RMSE_m['Climate_zone'].isin([data_type])].data.tolist(),
                      KGE_m[KGE_m['Climate_zone'].isin([data_type])].data.tolist()]
            p = position[t]
            bplot = axes[i].boxplot(t_data, patch_artist=True, positions=(p, p + 1, p + 2),
                                    widths=0.5, labels=labels, medianprops={'color': 'black', 'linewidth': '2.0'})
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        axes[i].set_xticks([i + 0.5 for i in position], data_types)
        axes[i].axhline(y=-0.4, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=(-0.5, 2))
        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=1, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set_ylabel(f'{model}', fontsize=18)
    axes[-1].set_axis_off()
    # axes[0].set_title(f'Train', fontsize=30, loc="left")
    # fig.legend(bplot['boxes'], labels, loc='lower center', ncol=3)
    axes[-1].legend(bplot['boxes'], labels, title=f'Train', loc='center', shadow=False, fontsize=30, title_fontsize=35)
    # plt.tight_layout()
    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case_name}_{model_name}_a4.eps", dpi=300)

def box_select(train_data, case, model_name, mm, y_lim,y):
    models = ['REA', 'GLEAM_hybrid', 'FLUXCOM']
    data_types = ['SAV', 'WSA', 'GRA', 'EBF', 'MF ', 'ENF', 'OSH', 'CRO', 'WET', 'DBF', 'CSH', 'BSV']

    fig, axes = plt.subplots(3, figsize=(16, 15))
    position = np.arange(1, 2 * len(data_types), 2)
    print(position)
    colors = sns.color_palette("Set3", n_colors=len(data_types), desat=.5).as_hex()

    for i, model in enumerate(models):
        model_data = train_data[train_data['models'].isin([model])]
        KGE_m = model_data[model_data['Metric'].isin([mm])]
        t_data = []
        for t, data_type in enumerate(data_types):
            t_data.append(KGE_m[KGE_m['Climate_zone'].isin([data_type])].data)
        bplot = axes[i].boxplot(t_data, patch_artist=True, positions=position,
                                widths=0.2, medianprops={'color': 'black', 'linewidth': '2.0'})
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for t, data_type in enumerate(data_types):
            df = KGE_m[KGE_m['Climate_zone'].isin([data_type])].data
            axes[i].text(position[t] + 0.2, y+0.1, f"{df.median():.3f}", fontsize=20, c='black')

        position1 = np.append(position, values=25)
        axes[i].set_xticks([i for i in position1], data_types + [''])
        axes[i].grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        axes[i].axhline(y=y, ls="--", c="black", alpha=0.7)  # 添加水平直线 #105885
        axes[i].set(ylim=y_lim)
        axes[i].set_ylabel(f'{model}')
    axes[0].set_title(f'Train', fontsize=30, loc="left")

    plt.savefig(f"/tera07/zhwei/For_QingChen/DataML/plot/fig6/fig6_{case}_{model_name}_{mm}_a.png", dpi=80)


if __name__ == "__main__":
    case_name = 'case3'
    model_name = 'model6'
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_lr_train.xlsx"
    train_data = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0
    train_data = train_data.dropna(subset=["data"])
    # box(train_data, case_name, model_name)


    mm = 'KGE'
    box_select(train_data, case_name, model_name, mm, (-0.25, 1),0.5)

    mm = 'R2'
    box_select(train_data, case_name, model_name, mm, (0, 1),0.5)

    mm = 'RMSE'
    box_select(train_data, case_name, model_name, mm, (0, 2),1)