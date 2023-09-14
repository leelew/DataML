import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook
import xarray as xr
import pandas as pd
import matplotlib.patches as mpatches
import math
from tqdm import tqdm, trange
import glob, os, shutil, sys
import seaborn as sns
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import subprocess
from pylab import rcParams
import argparse
from config import get_args
from pathlib import PosixPath, Path

font = {'family': 'Times new roman'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'grid.linewidth': 0.2,
          'font.size': 20,
          'legend.fontsize': 30,
          'legend.frameon': False,
          'xtick.labelsize': 32,
          'xtick.direction': 'out',
          'ytick.labelsize': 32,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)


def r2_score(y_true, y_pred):
    """
    计算R2分数
    """
    ssr = np.sum((y_pred - y_true.mean()) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    R2 = ssr / sst
    print(ssr, sst)
    print(y_true, y_pred)
    return R2


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


def rmse_score(y_true, y_pred):
    """
    计算RMSE分数
    """
    mse = np.mean((y_pred - y_true) ** 2)
    RMSE = np.sqrt(mse)
    return RMSE


def kge_score(y_true, y_pred):
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


def site_data_ML(cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    stnlist = f"{cfg['excle_path']}/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name=f'{test_set}')  # ,header=0
    data = pd.read_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_{test_set}.xlsx", sheet_name=f'{test_set}', header=0)
    pathout = f'{cfg["timeseries_path"]}/{case_name}_{model_name}/{test_set}'
    if os.path.exists(pathout):
        pass
    else:
        subprocess.run(f'mkdir {pathout}', shell=True)
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f"{cfg['input_path']}/{case_name}/model/{test_set}/{station_list['filename'][i]}.npy"
        ET = np.load(indata)[:, -1]
        ds = xr.open_dataset(f'{cfg["output_path"]}/{case_name}/{model_name}/{test_set}/{station_list["filename"][i]}.nc')
        ds = ds.rename({'n': 'time'})
        byear = station_list['Syear'][i]
        eyear = station_list['Eyear'][i]

        xtimes = pd.date_range("%s-01-01" % (byear), "%s-12-31" % (eyear), freq="1D")
        mtime = xr.DataArray(xtimes, coords={"time": xtimes}, dims=["time"])
        test = xr.DataArray(np.array(ET), coords={"time": xtimes}, dims=["time"])
        y_test = test.where(test > -100, drop=True)

        ds['time'] = y_test.time
        ds = ds.reindex(time=mtime.time)

        file = data[data['filename'].isin([station_list['filename'][i]])]

        fig, axes = plt.subplots(2, 2, figsize=(40, 15), sharey=True, sharex=True)

        models3 = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
        mmodel = ['mdl', 'gbm', 'am', 'RF']

        for m, model in enumerate(models3):
            j = np.tile(np.arange(0, 2), 2)[m]
            k = np.arange(0, 2).repeat(2)[m]
            mdata = file[file['models'].isin([model])]
            mlabel = 'FLUXNET'
            label = f"\nR2:{mdata[mdata['Metric'].isin(['R2'])].data.values[0]:.3f}" \
                    f"\nMSE:{mdata[mdata['Metric'].isin(['MSE'])].data.values[0]:.3f}" \
                    f"\nRMSE:{mdata[mdata['Metric'].isin(['RMSE'])].data.values[0]:.3f}" \
                    f"\nKGE: {mdata[mdata['Metric'].isin(['KGE'])].data.values[0]:.3f}"
            ds.et.plot.line(x='time', ax=axes[j][k], color='black', linewidth=1.5, linestyle="solid", alpha=0.8, label=f'{mlabel}')
            ds[f'{mmodel[m]}'].plot.line(x='time', ax=axes[j][k], linewidth=1.5, linestyle="--", alpha=0.8, label=f'{model} {label}')
            axes[j][k].legend(loc='best', shadow=False, frameon=False, fontsize=23)  # ,color=colors

        plt.tight_layout()
        plt.savefig(f"{pathout}/{station_list['filename'][i]}.png", dpi=300)
        plt.close()


def time_data_ML(cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    stnlist = f"{cfg['excle_path']}/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name=f'time_test')  # ,header=0
    train_data = pd.read_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_train_Timetest.xlsx", sheet_name=f'train_Timetest', header=0)
    test_data = pd.read_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_test_Timetest.xlsx", sheet_name=f'test_Timetest', header=0)
    pathout = f'{cfg["timeseries_path"]}/{case_name}_{model_name}/{test_set}/'
    if os.path.exists(pathout):
        pass
    else:
        subprocess.run(f'mkdir {pathout}', shell=True)
    split_ratio = 0.8
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = glob.glob(f"{cfg['input_path']}/{case_name}/model/Timetest/{station_list['filename'][i]}.npy")[0]
        ET = np.load(indata)[:, -1]
        byear = station_list['Syear'][i]
        eyear = station_list['Eyear'][i]

        xtimes = pd.date_range("%s-01-01" % (byear), "%s-12-31" % (eyear), freq="1D")
        mtime = xr.DataArray(xtimes, coords={"time": xtimes}, dims=["time"])
        ET_all = xr.DataArray(np.array(ET), coords={"time": xtimes}, dims=["time"])
        ET_data = ET_all.where(ET_all > -10, drop=True)
        N = int(split_ratio * len(ET_data))
        y_train = ET_data[:N]
        y_test = ET_data[N:]


        train = xr.open_dataset(f'{cfg["output_path"]}/{case_name}/{model_name}/train_Timetest/{station_list["filename"][i]}.nc').rename({'n': 'time'})
        train['time'] = y_train.time
        train = train.reindex(time=mtime.time)
        test = xr.open_dataset(f'{cfg["output_path"]}/{case_name}/{model_name}/test_Timetest/{station_list["filename"][i]}.nc').rename({'n': 'time'})
        test['time'] = y_test.time
        test = test.reindex(time=mtime.time)

        file_train = train_data[train_data['filename'].isin([station_list['filename'][i]])]
        file_test = test_data[test_data['filename'].isin([station_list['filename'][i]])]

        fig, axes = plt.subplots(2, 2, figsize=(40, 15))

        models3 = ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
        mmodel = ['mdl', 'gbm', 'am', 'RF']

        colors  = sns.color_palette("Set3", n_colors=5, desat=.9).as_hex()

        for m, model in enumerate(models3):
            j = np.tile(np.arange(0, 2), 2)[m]
            k = np.arange(0, 2).repeat(2)[m]
            mlabel = 'FLUXNET'
            mdata_train = file_train[file_train['models'].isin([model])]
            mdata_test = file_test[file_test['models'].isin([model])]
            label_train = f"\nR2:{mdata_train[mdata_train['Metric'].isin(['R2'])].data.values[0]:.3f}" \
                          f"\nMSE:{mdata_train[mdata_train['Metric'].isin(['MSE'])].data.values[0]:.3f}" \
                          f"\nRMSE:{mdata_train[mdata_train['Metric'].isin(['RMSE'])].data.values[0]:.3f}" \
                          f"\nKGE: {mdata_train[mdata_train['Metric'].isin(['KGE'])].data.values[0]:.3f}"
            label_test = f"\nR2:{mdata_test[mdata_test['Metric'].isin(['R2'])].data.values[0]:.3f}" \
                         f"\nMSE:{mdata_test[mdata_test['Metric'].isin(['MSE'])].data.values[0]:.3f}" \
                         f"\nRMSE:{mdata_test[mdata_test['Metric'].isin(['RMSE'])].data.values[0]:.3f}" \
                         f"\nKGE: {mdata_test[mdata_test['Metric'].isin(['KGE'])].data.values[0]:.3f}"
            ET_all.plot.line(x='time', ax=axes[j][k], color='black', linewidth=1.5, linestyle="solid", alpha=0.8, label=f'{mlabel}')
            train[f'{mmodel[m]}'].plot.line(x='time', ax=axes[j][k], linewidth=1.5, linestyle="--", alpha=0.8, label=f'{model} {label_train}',color=colors[4])
            test[f'{mmodel[m]}'].plot.line(x='time', ax=axes[j][k], linewidth=1.5, linestyle="--", alpha=0.8, label=f'{model} {label_test}',color=colors[3])
            axes[j][k].legend(loc='best', shadow=False, frameon=False, fontsize=23,ncol=3)  # ,color=colors

        plt.tight_layout()
        plt.savefig(f"{pathout}/{station_list['filename'][i]}.png", dpi=300)
        plt.close()


def site_data_select(cfg):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0
    train_data = pd.read_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case}_{filename}_M.xlsx", sheet_name='train', header=0)
    pathout = f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/fig/check/{case}/{model_name}/train/'

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f"/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/case3/{model_name}/train/{station_list['filename'][i]}.npy"
        ET = np.load(indata)[:, -1]

        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/case3/{model_name}/train/{station_list["filename"][i]}.nc')
        ds = ds.rename({'n': 'time'})
        byear = station_list['Syear'][i]
        eyear = station_list['Eyear'][i]

        xtimes = pd.date_range("%s-01-01" % (byear), "%s-12-31" % (eyear), freq="1D")
        mtime = xr.DataArray(xtimes, coords={"time": xtimes}, dims=["time"])
        train = xr.DataArray(np.array(ET), coords={"time": xtimes}, dims=["time"])
        y_train = train.where(train > -10, drop=True)

        ds['time'] = y_train.time
        ds = ds.reindex(time=mtime.time)

        models = ['REA', 'GLEAM_hybrid', 'FLUXCOM_9km']
        for j, model in enumerate(models):
            path = '/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/ET/'
            filename = glob.glob(f"{path}{model}/{station_list['filename'][i][:6]}*.nc", recursive=True)
            data = xr.open_dataset(filename[0]).ET
            ds[f'{model}'] = data

        file = train_data[train_data['ID'].isin([station_list['filename'][i]])]

        fig, axes = plt.subplots(2, 3, figsize=(60, 20))
        models1 = ['REA', 'GLEAM_hybrid', 'FLUXCOM_9km']
        models3 = ['DNN', 'LightGBM', 'Random Forest']
        mmodel = ['mdl', 'gbm', 'RF']
        colors = ['#94ccc2', '#f7f7bb', '#ed8b80', '#88b0cb', '#edb372', '#b0d275',
                  '#f7d2e5']  # sns.color_palette("Set3", n_colors=8, desat=.8).as_hex()
        # print(colors)

        for j, model in enumerate(models1):
            mdata = file[file['models'].isin([model])]
            label = f"\nR2:{mdata.R2.values} \nRMSE:{mdata.RMSE.values} \nKGE: {mdata.KGE.values}"
            ds.et.plot.line(x='time', ax=axes[0][j], color='black', linewidth=1.5, linestyle="solid", alpha=0.7, label='FLUXNET')
            model_p = ds[f'{model}']
            if (j == 2) & ((np.array(model_p.values) > 0).any()):
                model_p = ds[f"{model}"].where(ds[f"{model}"] > -99, drop=True)
            model_p.plot.line(x='time', ax=axes[0][j], linewidth=1.5, linestyle="--", label=f'{model} {label}', color=colors[j])
            axes[0][j].legend(loc='best', shadow=False, frameon=False, fontsize=30)  # ,color=colors

        for j, model in enumerate(models3):
            mdata = file[file['models'].isin([model])]
            label = f"\nR2:{mdata.R2.values} \nRMSE:{mdata.RMSE.values} \nKGE: {mdata.KGE.values}"
            ds.et.plot.line(x='time', ax=axes[1][j], color='black', linewidth=1.5, linestyle="solid", alpha=0.7, label='FLUXNET')
            ds[f'{mmodel[j]}'].plot.line(x='time', ax=axes[1][j], linewidth=1.5, linestyle="--", label=f'{model} {label}', color=colors[j + 3])
            axes[1][j].legend(loc='best', shadow=False, frameon=False, fontsize=30)  # ,color=colors

        plt.tight_layout()
        plt.savefig(f"{pathout}/{station_list['filename'][i]}.png", dpi=80)
        plt.close()


def time_data_select(cfg):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='test')  # ,header=0
    test_data = pd.read_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case}_{filename}_M.xlsx", sheet_name='test', header=0)
    pathout = f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/fig/check/{case}/{model_name}/test/'

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f"/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/case3/{model_name}/test/{station_list['filename'][i]}.npy"
        ET = np.load(indata)[:, -1]

        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/case3/{model_name}/test/{station_list["filename"][i]}.nc')
        ds = ds.rename({'n': 'time'})
        byear = station_list['Syear'][i]
        eyear = station_list['Eyear'][i]

        xtimes = pd.date_range("%s-01-01" % (byear), "%s-12-31" % (eyear), freq="1D")
        mtime = xr.DataArray(xtimes, coords={"time": xtimes}, dims=["time"])
        test = xr.DataArray(np.array(ET), coords={"time": xtimes}, dims=["time"])
        y_test = test.where(test > -10, drop=True)

        ds['time'] = y_test.time
        ds = ds.reindex(time=mtime.time)

        # models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
        # 'PMLV2']
        models = ['REA', 'GLEAM_hybrid', 'FLUXCOM_9km']
        for j, model in enumerate(models):
            path = '/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/ET/'
            filename = glob.glob(f"{path}{model}/{station_list['filename'][i][:6]}*.nc", recursive=True)
            data = xr.open_dataset(filename[0]).ET
            ds[f'{model}'] = data

        file = test_data[test_data['ID'].isin([station_list['filename'][i]])]

        fig, axes = plt.subplots(2, 3, figsize=(60, 20))
        models1 = ['REA', 'GLEAM_hybrid', 'FLUXCOM_9km']
        models3 = ['DNN', 'LightGBM', 'Random Forest']
        mmodel = ['mdl', 'gbm', 'RF']
        colors = ['#94ccc2', '#f7f7bb', '#ed8b80', '#88b0cb', '#edb372', '#b0d275',
                  '#f7d2e5']  # sns.color_palette("Set3", n_colors=8, desat=.8).as_hex()

        for j, model in enumerate(models1):
            mdata = file[file['models'].isin([model])]
            label = f"\nR2:{mdata.R2.values} \nRMSE:{mdata.RMSE.values} \nKGE: {mdata.KGE.values}"
            ds.et.plot.line(x='time', ax=axes[0][j], color='black', linewidth=1.5, linestyle="solid", alpha=0.7, label='FLUXNET')
            model_p = ds[f'{model}']
            if (j == 2) & ((np.array(model_p.values) > 0).any()):
                model_p = ds[f"{model}"].where(ds[f"{model}"] > -99, drop=True)
            model_p.plot.line(x='time', ax=axes[0][j], linewidth=1.5, linestyle="--", label=f'{model} {label}', color=colors[j])
            axes[0][j].legend(loc='best', shadow=False, frameon=False, fontsize=30)  # ,color=colors

        for j, model in enumerate(models3):
            mdata = file[file['models'].isin([model])]
            label = f"\nR2:{mdata.R2.values} \nRMSE:{mdata.RMSE.values} \nKGE: {mdata.KGE.values}"
            ds.et.plot.line(x='time', ax=axes[1][j], color='black', linewidth=1.5, linestyle="solid", alpha=0.7, label='FLUXNET')
            ds[f'{mmodel[j]}'].plot.line(x='time', ax=axes[1][j], linewidth=1.5, linestyle="--", label=f'{model} {label}', color=colors[j + 3])
            axes[1][j].legend(loc='best', shadow=False, frameon=False, fontsize=30)  # ,color=colors

        plt.tight_layout()
        plt.savefig(f"{pathout}/{station_list['filename'][i]}.png", dpi=80)
        plt.close()


def call_fun_by_str(cfg):
    if cfg['selected'] == 'true':
        if cfg['test_case'] == 'site test':
            eval('site_data_select')(cfg)
        if cfg['test_case'] == 'time test':
            eval('time_data_select')(cfg)
    else:
        if cfg['test_case'] == 'site test':
            eval('site_data_ML')(cfg)
        if cfg['test_case'] == 'time test':
            eval('time_data_ML')(cfg)


if __name__ == "__main__":
    cfg = get_args()
    call_fun_by_str(cfg)
    # case_name = 'case3'
    # model_name = 'model6'
    # # test_data_ML(case_name,model_name)
    # # train_data_ML(case_name,model_name)
    #
    # test_data_select(case_name, model_name, 'model6_lr')
    # train_data_select(case_name, model_name, 'model6_lr')
