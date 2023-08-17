import math
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import glob, os, shutil, sys
from tqdm import tqdm, trange
from sklearn.metrics import r2_score, mean_squared_error
import argparse
from config import get_args
from pathlib import PosixPath, Path


def R2_score(o, s):
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

def mse_score(y_true, y_pred):
    """
    计算RMSE分数
    """
    mse = np.mean((y_pred - y_true) ** 2)
    # mse = mean_squared_error(y_true, y_pred)
    return mse

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


def train_data(case_name, model_name):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0

    r2 = np.full((15, len(station_list['i'])), np.nan)
    rmse = np.full((15, len(station_list['i'])), np.nan)
    kge = np.full((15, len(station_list['i'])), np.nan)
    Climate_zone = []
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
              'PMLV2']
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f"/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/{case_name}/{model_name}/train/{station_list['filename'][i]}.npy"
        ET = np.load(indata)[:, -1]
        os.chdir("/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/ET/")
        for j, model in enumerate(models):
            filename = glob.glob(f"./{model}/{station_list['filename'][i]}*.nc", recursive=True)
            data = np.array(xr.open_dataset(filename[0]).ET)
            obs = np.array(ET)
            if np.count_nonzero(~np.isnan(data)) < len(data):
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(data)), axis=0)
                data = np.delete(np.array(data), np.argwhere(np.isnan(data)), axis=0)
            if np.count_nonzero(~np.isnan(obs)) < len(obs):
                data = np.delete(np.array(data), np.argwhere(np.isnan(obs)), axis=0)
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(obs)), axis=0)

            if len(obs) == 0:
                r2[j, i] = np.nan
                rmse[j, i] = np.nan
                kge[j, i] = np.nan
            else:
                r2[j, i] = corrlation_R2(obs, data)
                rmse[j, i] = rmse_score(obs, data)
                kge[j, i] = kge_score(obs, data)

        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/train/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        r2[11:, i] = np.array(
            [corrlation_R2(ET_sim, mdl_sim), corrlation_R2(ET_sim, lgb_sim), corrlation_R2(ET_sim, am_sim), corrlation_R2(ET_sim, RF_sim)])
        rmse[11:, i] = np.array([rmse_score(ET_sim, mdl_sim), rmse_score(ET_sim, lgb_sim), rmse_score(ET_sim, am_sim), rmse_score(ET_sim, RF_sim)])
        kge[11:, i] = np.array([kge_score(ET_sim, mdl_sim), kge_score(ET_sim, lgb_sim), kge_score(ET_sim, am_sim), kge_score(ET_sim, RF_sim)])

        Climate_zone.append(station_list['Climate_zone'][i])

    models = models + ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    print(models)
    train_data = pd.DataFrame({'ID': np.tile(station_list['filename'], 15),
                               'models': np.array(models).repeat(len(station_list['filename'])),
                               'R2': r2.reshape(-1),
                               'RMSE': rmse.reshape(-1),
                               'KGE': kge.reshape(-1),
                               'Climate_zone': np.tile(Climate_zone, len(models))})  # 'R': R,'KGE': KGE,
    # train_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_train_M.xlsx", sheet_name='train', index=True)
    train_data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 3),
                               'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                               'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                               'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                               'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    train_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_lr_train.xlsx", sheet_name='train', index=True)


def test_data(case_name, model_name):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='test')  # ,header=0

    r2 = np.full((15, len(station_list['i'])), np.nan)
    rmse = np.full((15, len(station_list['i'])), np.nan)
    kge = np.full((15, len(station_list['i'])), np.nan)
    models = ['GLEAM_v3.6a', 'GLEAM_v3.6b', 'GLEAM_hybrid', 'REA', 'GLDAS_CLSM-2.2', 'GLDAS_Noah-2.1', 'ERA5', 'ET_3T', 'EB_ET', 'FLUXCOM_9km',
              'PMLV2']
    Climate_zone = []

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        indata = f"/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/{case_name}/{model_name}/test/{station_list['filename'][i]}.npy"
        ET = np.load(indata)[:, -1]
        os.chdir("/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/ET/")
        for j, model in enumerate(models):
            filename = glob.glob(f"./{model}/{station_list['filename'][i][:6]}*.nc", recursive=True)
            data = np.array(xr.open_dataset(filename[0]).ET)
            obs = np.array(ET)
            if np.count_nonzero(~np.isnan(data)) < len(data):
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(data)), axis=0)
                data = np.delete(np.array(data), np.argwhere(np.isnan(data)), axis=0)
            if np.count_nonzero(~np.isnan(obs)) < len(obs):
                data = np.delete(np.array(data), np.argwhere(np.isnan(obs)), axis=0)
                obs = np.delete(np.array(obs), np.argwhere(np.isnan(obs)), axis=0)
            if len(obs) > 0:
                r2[j, i] = corrlation_R2(obs, data)
                rmse[j, i] = rmse_score(obs, data)
                kge[j, i] = kge_score(obs, data)



        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/test/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        r2[11:, i] = np.array(
            [corrlation_R2(ET_sim, mdl_sim), corrlation_R2(ET_sim, lgb_sim), corrlation_R2(ET_sim, am_sim), corrlation_R2(ET_sim, RF_sim)])
        rmse[11:, i] = np.array([rmse_score(ET_sim, mdl_sim), rmse_score(ET_sim, lgb_sim), rmse_score(ET_sim, am_sim), rmse_score(ET_sim, RF_sim)])
        kge[11:, i] = np.array([kge_score(ET_sim, mdl_sim), kge_score(ET_sim, lgb_sim), kge_score(ET_sim, am_sim), kge_score(ET_sim, RF_sim)])
        Climate_zone.append(station_list['Climate_zone'][i])

    models = models + ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    print(models)
    test_data = pd.DataFrame({'ID': np.tile(station_list['filename'], 15),
                              'models': np.array(models).repeat(len(station_list['filename'])),
                              'R2': r2.reshape(-1),
                              'RMSE': rmse.reshape(-1),
                              'KGE': kge.reshape(-1),
                              'Climate_zone': np.tile(Climate_zone, len(models))})  # 'R': R,'KGE': KGE,
    # test_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_test_M.xlsx", sheet_name='test', index=True)

    test_data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 3),
                              'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                              'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                              'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                              'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    test_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_lr_test.xlsx", sheet_name='test', index=True)


def train_data_ensmean(case_name, model_name):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='train')  # ,header=0

    r2 = np.full((len(station_list['i'])), np.nan)
    rmse = np.full((len(station_list['i'])), np.nan)
    kge = np.full((len(station_list['i'])), np.nan)
    Climate_zone = []
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/train/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)

        ET_obs = np.mean([mdl_sim, lgb_sim, am_sim, RF_sim], axis=0)
        r2[i] = np.array(corrlation_R2(ET_sim, ET_obs))
        rmse[i] = np.array(rmse_score(ET_sim, ET_obs))
        kge[i] = np.array(kge_score(ET_sim, ET_obs))

        Climate_zone.append(station_list['Climate_zone'][i])

    models = ['ensmean']
    train_data = pd.DataFrame({'ID': np.tile(station_list['filename'], len(models)),
                               'models': np.array(models).repeat(len(station_list['filename'])),
                               'R2': r2.reshape(-1),
                               'RMSE': rmse.reshape(-1),
                               'KGE': kge.reshape(-1),
                               'Climate_zone': np.tile(Climate_zone, len(models))})  # 'R': R,'KGE': KGE,
    train_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_train_ens_M.xlsx", sheet_name='train', index=True)
    train_data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 3),
                               'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                               'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                               'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                               'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    train_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_train_ens.xlsx", sheet_name='train', index=True)


def test_data_ensmean(case_name, model_name):
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name='test')  # ,header=0

    r2 = np.full((len(station_list['i'])), np.nan)
    rmse = np.full((len(station_list['i'])), np.nan)
    kge = np.full((len(station_list['i'])), np.nan)
    Climate_zone = []

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))

        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/test/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)

        ET_obs = np.mean([mdl_sim, lgb_sim, am_sim, RF_sim], axis=0)
        r2[i] = np.array(corrlation_R2(ET_sim, ET_obs))
        rmse[i] = np.array(rmse_score(ET_sim, ET_obs))
        kge[i] = np.array(kge_score(ET_sim, ET_obs))

        Climate_zone.append(station_list['Climate_zone'][i])
    models = ['ensmean']
    test_data = pd.DataFrame({'ID': np.tile(station_list['filename'],len(models)),
                              'models': np.array(models).repeat(len(station_list['filename'])),
                              'R2': r2.reshape(-1),
                              'RMSE': rmse.reshape(-1),
                              'KGE': kge.reshape(-1),
                              'Climate_zone': np.tile(Climate_zone, len(models))})  # 'R': R,'KGE': KGE,
    test_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_test_ens_M.xlsx", sheet_name='test', index=True)

    test_data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 3),
                              'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 3),
                              'Metric': np.array(['R2', 'RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                              'Climate_zone': np.tile(Climate_zone, len(models) * 3),
                              'data': np.concatenate([r2.reshape(-1), rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    test_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_test_ens.xlsx", sheet_name='test', index=True)

def site_data_ML(cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name=f'{test_set}')  # ,header=0

    r2 = np.full((4, len(station_list['i'])), np.nan)
    mse = np.full((4, len(station_list['i'])), np.nan)
    rmse = np.full((4, len(station_list['i'])), np.nan)
    kge = np.full((4, len(station_list['i'])), np.nan)
    Climate_zone = []
    #
    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))

        ds = xr.open_dataset(f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/{test_set}/%s.nc' % (station_list['filename'][i]))
        ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
        mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
        r2[:, i] = np.array([R2_score(ET_sim, mdl_sim), R2_score(ET_sim, lgb_sim), R2_score(ET_sim, am_sim), R2_score(ET_sim, RF_sim)])
        mse[:, i] = np.array([mse_score(ET_sim, mdl_sim), mse_score(ET_sim, lgb_sim), mse_score(ET_sim, am_sim), mse_score(ET_sim, RF_sim)])
        rmse[:, i] = np.array([rmse_score(ET_sim, mdl_sim), rmse_score(ET_sim, lgb_sim), rmse_score(ET_sim, am_sim), rmse_score(ET_sim, RF_sim)])
        kge[:, i] = np.array([kge_score(ET_sim, mdl_sim), kge_score(ET_sim, lgb_sim), kge_score(ET_sim, am_sim), kge_score(ET_sim, RF_sim)])

        Climate_zone.append(station_list['Climate_zone'][i])

    models =  ['DNN', 'LightGBM', 'AutoML', 'Random Forest']
    # train_data = pd.DataFrame({'ID': np.tile(station_list['filename'], 4),
    #                            'models': np.array(models).repeat(len(station_list['filename'])),
    #                            'R2': r2.reshape(-1),
    #                            'RMSE': rmse.reshape(-1),
    #                            'KGE': kge.reshape(-1),
    #                            'Climate_zone': np.tile(Climate_zone, len(models))})  # 'R': R,'KGE': KGE,
    # # train_data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_train_M.xlsx", sheet_name='train', index=True)
    data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 4),
                          'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 4),
                          'Metric': np.array(['R2', 'MSE','RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                          'Climate_zone': np.tile(Climate_zone, len(models) * 4),
                          'data': np.concatenate([r2.reshape(-1), mse.reshape(-1),rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_{test_set}.xlsx", sheet_name=f'{test_set}', index=True)


def time_data_ML(cfg):
    case_name = cfg['case_name']
    model_name = cfg['model_name']
    test_set = cfg['testset']
    stnlist = f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}.xlsx"
    station_list = pd.read_excel(stnlist, header=0, sheet_name=f'train_test')  # ,header=0

    r2 = np.full((4, len(station_list['i'])), np.nan)
    mse = np.full((4, len(station_list['i'])), np.nan)
    rmse = np.full((4, len(station_list['i'])), np.nan)
    kge = np.full((4, len(station_list['i'])), np.nan)

    Climate_zone = []

    pbar = tqdm(range(len(station_list['filename'])), ncols=140)
    for i in pbar:
        pbar.set_description("Now loacation at %s" % (station_list['filename'][i]))
        filevar = f'/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/{case_name}/{model_name}/{test_set}/%s.nc' % (station_list['filename'][i])
        if os.path.exists(filevar):
            ds = xr.open_dataset(filevar)
            ET_sim, lgb_sim, mdl_sim, am_sim, RF_sim = ds.et.values, ds.gbm.values, ds.mdl.values, ds.am.values, ds.RF.values
            mdl_sim = np.delete(mdl_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
            lgb_sim = np.delete(lgb_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
            am_sim = np.delete(am_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
            RF_sim = np.delete(RF_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
            ET_sim = np.delete(ET_sim, np.argwhere(np.isnan(ET_sim)), axis=0)
            r2[:, i] = np.array([R2_score(ET_sim, mdl_sim), R2_score(ET_sim, lgb_sim), R2_score(ET_sim, am_sim), R2_score(ET_sim, RF_sim)])
            mse[:, i] = np.array([mse_score(ET_sim, mdl_sim), mse_score(ET_sim, lgb_sim), mse_score(ET_sim, am_sim), mse_score(ET_sim, RF_sim)])
            rmse[:, i] = np.array([rmse_score(ET_sim, mdl_sim), rmse_score(ET_sim, lgb_sim), rmse_score(ET_sim, am_sim), rmse_score(ET_sim, RF_sim)])
            kge[:, i] = np.array([kge_score(ET_sim, mdl_sim), kge_score(ET_sim, lgb_sim), kge_score(ET_sim, am_sim), kge_score(ET_sim, RF_sim)])
        Climate_zone.append(station_list['Climate_zone'][i])

    models =['DNN', 'LightGBM', 'AutoML', 'Random Forest']

    data = pd.DataFrame({'filename': np.tile(station_list['filename'], len(models) * 4),
                          'models': np.tile(np.array(models).repeat(len(station_list['filename'])), 4),
                          'Metric': np.array(['R2', 'MSE','RMSE', 'KGE']).repeat(len(station_list['filename']) * len(models)),
                          'Climate_zone': np.tile(Climate_zone, len(models) * 4),
                          'data': np.concatenate([r2.reshape(-1), mse.reshape(-1),rmse.reshape(-1), kge.reshape(-1)], axis=0)})  # 'R': R,'KGE': KGE,
    data.to_excel(f"/tera07/zhwei/For_QingChen/DataML/plot/xlsx/{case_name}_{model_name}_{test_set}.xlsx", sheet_name=f'{test_set}', index=True)

def call_fun_by_str(cfg):
    if os.path.exists(f"{cfg['case_name']}_{cfg['model_name']}_{cfg['testset']}.xlsx"):
        print(f"{cfg['case_name']}_{cfg['model_name']}_{cfg['testset']}.xlsx exist")
    else:
        if cfg['test_case'] == 'site test':
            eval('site_data_ML')(cfg)
        if cfg['test_case'] == 'time test':
            eval('time_data_ML')(cfg)

if __name__ == "__main__":
    cfg = get_args()
    call_fun_by_str(cfg)
    # case_name = 'case5'
    # model_name = 'model2'
    # train_data(case_name, model_name)
    # test_data(case_name, model_name)
    # train_data_ensmean(case_name, model_name)
    # test_data_ensmean(case_name, model_name)
    # train_data_ML(cfg)
    # test_data_ML(cfg)
