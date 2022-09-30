import argparse
import pickle
from pathlib import PosixPath, Path
import time
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import xarray as xr
import netCDF4 as nc

from data import Dataset
from config import get_args


def main(cfg):
    print("Now we training {et} product in {tr} temporal resolution and {sr} spatial resolution with ERA5-Land forcing, Yuan et al. LAI, climate zone, DEM, land cover, SoilGrids, Hope you'll get the satisfied result!".format(et=cfg["et_product"],tr=cfg["temporal_resolution"],sr=cfg["spatial_resolution"]))

    print('[DataML] Make & load inputs')
    path = cfg["inputs_path"]+cfg["et_product"]
    if os.path.exists(path+'x_train.npy'):
        print(' [DataML] loading input data')
        x_train = np.load(path+'x_train.npy')   
        y_train = np.load(path+'y_train.npy')
        x_test = np.load(path+'x_test.npy')
        y_test = np.load(path+'y_test.npy')
    else:     
        # load data
        print(' [DataML] making input data')
        cls = Dataset(cfg) #FIXME: saving to input path
        x_train, y_train, x_test, y_test, lat, lon = cls.fit()


    print('[DataML] Train & load LightGBM')
    path = cfg["outputs_path"]+'saved_model/'+cfg["et_product"]+'/'
    if os.path.exists(path+'model.pickle'):
        # load model
        print('[dataML] loading trained model') 
        f = open(path+'model.pickle','rb')
        gbm = pickle.load(f)
    else:
        # train 
        print('[dataML] training LightGBM') 
        gbm = lgb.LGBMRegressor(
            objective='regression', 
            num_leaves=cfg["num_leaves"], 
            n_estimators=cfg["n_estimators"])
        gbm.fit(x_train, y_train)
        
        # save model
        print('[dataML] saving LightGBM') 
        output = open(path+'model.pickle','wb')
        pickle.dump(gbm, output)
        output.close()
    
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    print('[DataML] Estimating')
    # pred
    y_pred = np.ones_like(y_test)*np.nan
    for i in range(len(lat)):
        for j in range(len(lon)):
            m,n = x_test[:,i,j], y_test[:,i,j]
            if not np.isnan(n).all():
                tmp = gbm.predict(m)
                y_pred[:,i,j] = tmp

    if cfg["normalize"]:
        with open('scaler.json', 'r') as f:
            scaler = json.load(f)
        y_pred = cls.reverse_normalize(y_pred, 'output', scaler, 'minmax', -1)

    # fit to monthly scale
    n = y_test.shape[0]//31
    y_pred_month = np.full((n,720,1440),np.nan)
    y_test_month = np.full((n,720,1440),np.nan)
    for i in range(n):
        y_test_month[i] = np.nanmean(y_test[31*i:31*i+32],axis=0)
        y_pred_month[i] = np.nanmean(y_pred[31*i:31*i+32],axis=0)

    # save pred
    path = cfg["outputs_path"]+"forecast/"+cfg["et_product"]
    name = 'ET_{name}_{tr}_{sr}_{begin_year}_{end_year}_LightGBM.nc'.format(
        name=cfg["et_product"],
        tr=cfg["temporal_resolution"],
        sr=cfg["spatial_resolution"],
        begin_year=cfg["begin_year"],
        end_year=cfg["end_year"])
    f = nc.Dataset(name, 'w', format='NETCDF4')
    f.createDimension('longitude', size=y_pred.shape[-1])
    f.createDimension('latitude', size=y_pred.shape[-2])
    f.createDimension('time', size=y_pred.shape[-3])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('ET', 'f4', dimensions=('time','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, y_pred
    f.close()
    os.system('mv {} {}'.format(name, path))


    name = 'ET_{name}_1M_{sr}_{begin_year}_{end_year}_LightGBM.nc'.format(
        name=cfg["et_product"],
        sr=cfg["spatial_resolution"],
        begin_year=cfg["begin_year"],
        end_year=cfg["end_year"])
    f = nc.Dataset(name, 'w', format='NETCDF4')
    f.createDimension('longitude', size=y_pred_month.shape[-1])
    f.createDimension('latitude', size=y_pred_month.shape[-2])
    f.createDimension('time', size=y_pred_month.shape[-3])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('ET', 'f4', dimensions=('time','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, y_pred_month
    f.close()
    os.system('mv {} {}'.format(name, path))


    # cal metrics
    print('[dataML] cal metrics, God bless U!') 
    r2 = np.full((len(lat),len(lon)),np.nan)
    rmse = np.full((len(lat),len(lon)),np.nan)
    r = np.full((len(lat),len(lon)),np.nan)

    r2_m = np.full((len(lat),len(lon)),np.nan)
    rmse_m = np.full((len(lat),len(lon)),np.nan)
    r_m = np.full((len(lat),len(lon)),np.nan)

    for i in range(len(lat)):
        for j in range(len(lon)):
            n,m,k,j = y_test[:,i,j],y_pred[:,i,j],y_test_month[:,i,j],y_pred_month[:,i,j]
            if not np.isnan(n).all():
                c = np.delete(m, np.where(np.isnan(n)),axis=0)
                d = np.delete(n, np.where(np.isnan(n)),axis=0)
                r2[i,j] = r2_score(d, c)
                rmse[i,j] = np.sqrt(mean_squared_error(d,c)) 
                r[i,j] = np.corrcoef(d,c)[0,1]

                c = np.delete(j, np.where(np.isnan(k)),axis=0)
                d = np.delete(k, np.where(np.isnan(k)),axis=0)
                r2_m[i,j] = r2_score(d, c)
                rmse_m[i,j] = np.sqrt(mean_squared_error(d,c)) 
                r_m[i,j] = np.corrcoef(d,c)[0,1]
    r2[r2<0] = 0
    r[r<0] = 0
    r2_m[r2_m<0] = 0
    r_m[r_m<0] = 0
    metric = np.stack([r2,r,rmse],axis=0)
    metric_m = np.stack([r2_m,r_m,rmse_m],axis=0)


    name = 'metrics_{name}_{tr}_{sr}_{begin_year}_{end_year}_LightGBM.nc'.format(
        name=cfg["et_product"],
        tr=cfg["temporal_resolution"],
        sr=cfg["spatial_resolution"],
        begin_year=cfg["begin_year"],
        end_year=cfg["end_year"])
    f = nc.Dataset(name, 'w', format='NETCDF4')
    f.createDimension('longitude', size=r2.shape[-1])
    f.createDimension('latitude', size=r2.shape[-2])
    f.createDimension('num', size=metric.shape[0])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('metric', 'f4', dimensions=('num','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, metric
    f.close()
    os.system('mv {} {}'.format(name, path))

    name = 'metrics_{name}_1M_{sr}_{begin_year}_{end_year}_LightGBM.nc'.format(
        name=cfg["et_product"],
        sr=cfg["spatial_resolution"],
        begin_year=cfg["begin_year"],
        end_year=cfg["end_year"])
    f = nc.Dataset(name, 'w', format='NETCDF4')
    f.createDimension('longitude', size=metric_m.shape[-1])
    f.createDimension('latitude', size=metric_m.shape[-2])
    f.createDimension('num', size=metric_m.shape[0])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('metric', 'f4', dimensions=('num','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, metric_m
    f.close()
    os.system('mv {} {}'.format(name, path))



if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
