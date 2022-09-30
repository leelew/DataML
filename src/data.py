# author: Lu Li
# mail: lilu83@mail.sysu.edu.cn

from email import message
from locale import normalize
import os
import glob
import json
import datetime

import numpy as np
import xarray as xr



class Dataset():
    __name__ = ['fit','clip_by_date']

    def __init__(self, cfg: dict):
        """_summary_

        Args:
            forcing_root (str): root path of forcing data;
            et_root (str): root path of target et data;
            lai_root (str): root path of lai data;
            ancillary_root (str): root path of ancillary data;
            et_product (str): product name of target et data;
            begin_year (int): begin year for training;
            end_year (int): end year for training;
            et_begin_year (int): begin year of target et data;
            et_end_year (int): end year of target et data;
            t_resolution (str): temporal resolution of target et data;
            s_resolution (str): spatial resolution of target et data;
            split_ratio (float): split ratio of training/test data;
            crop (bool): if or not crop sub-regions
            normalize (bool): if or not normalization
        """
        self.split_ratio = cfg["split_ratio"]
        self.et_product = cfg["et_product"]
        self.begin_year = cfg["begin_year"]
        self.end_year = cfg["end_year"]
        self.et_begin_year = cfg["et_begin_year"]
        self.et_end_year = cfg["et_end_year"]
        self.t_resolution = cfg["temporal_resolution"]
        self.s_resolution = cfg["spatial_resolution"]
        self.crop = cfg["crop"] #temporary params
        self.normalize = cfg["normalize"]
        self.inputs_path = cfg["inputs_path"]
        self.data_path = cfg["data_path"]


        # check the select begin/end year
        # NOTE: selected begin/end year should between 
        #       begin/end year of target et data. if the selected year 
        #       out of the range, then fit to the begin/end year.
        if self.et_end_year<self.end_year:
            message = 'clip end year {end_year} > {product} {et_end_year}'.format(
                end_year=self.end_year,
                product=self.et_product,
                et_end_year=self.et_end_year)
            print(message)
            self.et_end_year = self.end_year
        if self.et_begin_year>self.begin_year:
            message = 'clip begin year {begin_year} < {product} {et_begin_year}'.format(
                end_year=self.begin_year,
                product=self.et_product,
                et_end_year=self.et_begin_year)
            print(message)
            self.et_begin_year = self.begin_year

        # forcing name for ERA5-Land.
        # NOTE: The name of data should be "ERA5Land_2001_P_1D_025D.nc"
        #       The variable name in netcdf file should be P.
        #       We also add swc into forcing for convenience.
        forcing_list = ['P','Q','sp','ssrd','strd','t2m','u10','v10','swvl1']
        self.forcing_list = forcing_list
        self.num_forcing = len(forcing_list)


    def fit(self):
        """main process for making training/test data"""
        # get path of target et data
        et_path = glob.glob(self.data_path+'ET/'+ "*{name}*nc".format(name=self.et_product))[0]

        PATH = self.inputs_path+self.et_product+'/'
        print('  [DataML] loading lat/lon grids')
        lat_file_name = 'lat_{t}_{s}.npy'.format(t=self.t_resolution,s=self.s_resolution)
        lon_file_name = 'lon_{t}_{s}.npy'.format(t=self.t_resolution,s=self.s_resolution)
        if os.path.exists(PATH+lat_file_name):
            lat,lon = np.load(PATH+lat_file_name),np.load(PATH+lon_file_name)
        else:
            with xr.open_dataset(et_path) as f:
                lat, lon = np.array(f.latitude), np.array(f.longitude)
            np.save(PATH+lat_file_name,lat)
            np.save(PATH+lon_file_name,lon)
    

        print('  [DataML] loading forcing')
        file_name = 'ERA5-Land_forcing_{tr}_{sr}_{begin_year}_{end_year}.npy'.format(
            tr=self.t_resolution, 
            sr=self.s_resolution, 
            begin_year=self.begin_year, 
            end_year=self.end_year)
    
        if os.path.exists(PATH+file_name):
            forcing = np.load(PATH+file_name) #(t,lat,lon,feat)
        else:
            fold = '{tr}_{sr}'.format(tr=self.t_resolution,sr=self.s_resolution)
            if not os.path.exists(self.data_path+'forcing/'+fold):
                message = ("ERA5-Land forcing on {tr} temporal resolution and {sr} spatial resolution doesn't exist, please make forcing before".format(
                    tr=self.t_resolution,sr=self.s_resolution))
                raise IOError(message)
            else:
                forcing = self._load_forcing(self.data_path+'forcing/', 
                                            self.forcing_list, 
                                            self.begin_year, 
                                            self.end_year,
                                            self.t_resolution,
                                            self.s_resolution)
                np.save(PATH+file_name, forcing)


        print('  [DataML] loading ET')
        file_name = '{product}_ET_{tr}_{sr}_{begin_year}_{end_year}.npy'.format(
            product=self.et_product, 
            tr=self.t_resolution, 
            sr=self.s_resolution,
            begin_year=self.begin_year, 
            end_year=self.end_year)
        
        if os.path.exists(PATH+file_name):
            et = np.load(PATH+file_name) #(t,lat,lon)
        else:
            et = self._load_et(
                self.data_path+'ET/', self.et_product, self.t_resolution, self.s_resolution)
            begin_idx, end_idx = self._clip_by_date(begin_date=self.et_begin_year, 
                                                   end_date=self.et_end_year, 
                                                   select_begin_date=self.begin_year, 
                                                   select_end_date=self.end_year)
            et = et[begin_idx:end_idx+1] #+1 for python
            np.save(PATH+file_name, et)


        print('  [DataML] loading LAI')
        file_name = 'MODIS_LAI_{tr}_{sr}_{begin_year}_{end_year}.npy'.format(
            tr=self.t_resolution, 
            sr=self.s_resolution,
            begin_year=self.begin_year, 
            end_year=self.end_year)

        if os.path.exists(PATH+file_name):
            lai = np.load(PATH+file_name) #(t,lat,lon)
        else:
            lai = self._load_lai(self.data_path+"LAI/")
            #FIXME(Lu Li): if select begin date < 2000, use 2000 or 2004 instead
            if self.begin_year<2000:
                raise IOError('Sorry, there is a bug for generating LAI data before 2000, please used begin year after 2000!')
            begin_idx, end_idx = self._clip_by_date(begin_date=2000, 
                                                   end_date=2021, 
                                                   select_begin_date=self.begin_year, 
                                                   select_end_date=self.end_year) 
            lai = lai[begin_idx:end_idx+1,:,:,np.newaxis] # for concat
            np.save(PATH+file_name, lai)


        print('  [DataML] loading ancillary')
        file_name = 'static_{tr}_{sr}.npy'.format(
            tr=self.t_resolution, 
            sr=self.s_resolution)

        if os.path.exists(self.inputs_path+file_name):
            static = np.load(self.inputs_path+file_name) #(n,lat,lon)
        else:
            static = self._load_ancillary(self.data_path+"ancillary/", self.s_resolution)
            np.save(self.inputs_path+file_name, static)


        print('begin:{begin_year}, end:{end_year}'.format(
            begin_year=self.begin_year, end_year=self.end_year))
        print('forcing shape is {shape}'.format(shape=forcing.shape))        
        print('ET shape is {shape}'.format(shape=et.shape))        
        print('LAI shape is {shape}'.format(shape=lai.shape))        
        print('static shape is {shape}'.format(shape=static.shape))   
        assert forcing.shape[0]==et.shape[0], "X(t) /= ET(t)"
        assert forcing.shape[0]==lai.shape[0], "X(t) /= LAI(t)"   
        # get shape
        self.time_length, self.nlat, self.nlon, self.num_features = forcing.shape
        N = int(self.split_ratio*self.time_length) 
        print('{n} samples for training, {m} samples for testing'.format(
            n=N,m=self.time_length-N))


        print('[DataML] preprocessing')
        static = np.tile(static,(self.time_length,1,1,1))
        feat = np.concatenate([forcing,lai,static],axis=-1)
        del forcing, lai, static

        x_train, y_train = feat[:N], et[:N]
        x_test, y_test = feat[N:], et[N:]
        del feat, et

        # NOTE: remove annual p<200mm/day grids to increase 
        #       the performance over the equator
        #       @suggestion by Zhongwang Wei
        #mask = np.zeros((self.nlat,self.nlon))*np.nan
        #p = forcing[:,:,:,0] # get precipitation
        #for i in range(720):
        #    for j in range(1440):
        #        tmp = np.nanmean(p[:,i,j])*365*1000*24 #m/hr -> mm/day
        #        if tmp > 200:
        #            mask[i,j] = 1    
        #lat_idx, lon_idx = np.where(mask==1)[0], np.where(mask==1)[1]
        #x_train = x_train[:,lat_idx,lon_idx,:]
        #y_train = y_train[:,lat_idx,lon_idx,:]
            

        # FIXME: Not completed!
        if self.normalize:
            print('[DataML] start normalization feat')
            scaler = self._get_minmax_scaler(x_train, y_train, 'region')
            with open('scaler.json', 'w') as f:
                json.dump(scaler, f)
            print(x_train.shape, y_train.shape, x_test.shape)
            x_train = self._normalize(x_train, 'input', scaler, 'minmax')
            y_train = self._normalize(y_train, 'output', scaler, 'minmax')
            x_test = self._normalize(x_test, 'input', scaler, 'minmax')

        # FIXME: add parameters for region boundary, or lat/lon list
        #        now only support for cropping amazon    
        if self.crop: 
            print('[DataML] start crop AMAZON, more flexiable crop mode will comes later!')
            lat_idx = np.where((lat>-15) & (lat<10))[0]
            lon_idx = np.where((lon>-82) & (lon<-33))[0]
            x_train = x_train[:,lat_idx][:,:,lon_idx]
            y_train = y_train[:,lat_idx][:,:,lon_idx]

        # save output
        x_train = x_train.reshape(-1, x_train.shape[-1])
        y_train = y_train.reshape(-1, 1)
        x_train = np.delete(x_train, np.argwhere(np.isnan(y_train)), axis=0)
        y_train = np.delete(y_train, np.argwhere(np.isnan(y_train)), axis=0)

        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
        np.save('y_test.npy', y_test)
        os.system('mv {} {}'.format("*.npy", PATH))

        print('{n} million feats for training'.format(n=x_train.shape[0]*x_train.shape[1]/1000000))
        print('{n} million samples for training'.format(n=x_train.shape[0]/1000000))
        return x_train, y_train, x_test, y_test, lat, lon


    def _load_forcing(self, 
                      forcing_root, 
                      forcing_list, 
                      begin_year, 
                      end_year, 
                      t_resolution, 
                      s_resolution):
        forcing = []
        for year in range(begin_year, end_year+1):
            tmp = []
            for i in range(self.num_forcing):
                fold = "{tr}_{sr}/".format(tr=t_resolution, sr=s_resolution)
                file = forcing_root + fold + "/ERA5Land_{year}_{var}_{tr}_{sr}.nc".format(
                    year=year, var=forcing_list[i], tr=t_resolution, sr=s_resolution)
                with xr.open_dataset(file) as f:
                    tmp.append(f[forcing_list[i]])
            tmp = np.stack(tmp, axis=-1)
            forcing.append(tmp)
        forcing = np.concatenate(forcing, axis=0)
        return forcing

    def _load_et(self, et_root, et_product, temporal_resolution, spatial_resolution):
        l = glob.glob(et_root + "*{name}_{tr}_{sr}*nc".format(name=et_product,tr=temporal_resolution, sr=spatial_resolution))
        with xr.open_dataset(l[0]) as f:
            return np.array(f.ET)

    def _load_lai(self, lai_root):
        with xr.open_dataset(lai_root+'LAI_1D_0p25_2000-2020.nc') as f:
            lai = np.array(f.LAI)
        return lai

    def _load_ancillary(self, ancillary_root, s_resolution):
        ancillary_root = ancillary_root + '/' + s_resolution + '/'
        with xr.open_dataset(ancillary_root+'Beck_KG_V1_present_{}.nc'.format(s_resolution)) as f:
            climate_zone = np.array(f.climate_zone)
        with xr.open_dataset(ancillary_root+'DEM_{}.nc'.format(s_resolution)) as f:
            dem = np.array(f.dem)
        with xr.open_dataset(ancillary_root+'kosugi_{}.nc'.format(s_resolution)) as f:
            kosugi = np.array(f.kosugi)
        with xr.open_dataset(ancillary_root+'LC_{}.nc'.format(s_resolution)) as f:
            land_cover = np.array(f.land_cover)
        with xr.open_dataset(ancillary_root+'soilgrid_{}.nc'.format(s_resolution)) as f:
            soilgrid = np.array(f.soilgrids)
        
        tmp = np.stack([climate_zone,dem,land_cover,soilgrid[0],soilgrid[1],soilgrid[7],soilgrid[8],soilgrid[14],soilgrid[15]], axis=0)
        static = np.concatenate([tmp, kosugi],axis=0)
        static = np.transpose(static,(1,2,0))[np.newaxis] # for concat
        return static

    def _normalize(self, feature, variable, scaler, scaler_type):
        if scaler_type == 'standard':
            if variable == 'input':
                feature = (feature - np.array(scaler["input_mean"])) / np.array(scaler["input_std"])
            elif variable == 'output':
                feature = (feature - np.array(scaler["output_mean"])) / np.array(scaler["output_std"])
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        elif scaler_type == 'minmax':
            if variable == 'input':
                feature = (feature - np.array(scaler["input_mean"])) / (np.array(scaler["input_std"])-np.array(scaler["input_mean"]))
            elif variable == 'output':
                feature = (feature - np.array(scaler["output_mean"])) / (np.array(scaler["output_std"])-np.array(scaler["output_mean"]))
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        return feature
    
    def reverse_normalize(
                        self,
                        feature,
                        variable: str,
                        scaler,
                        scaler_method: str,
                        is_multivars: int) -> np.ndarray:
        """reverse normalized features using pre-computed statistics"""
        a, b = np.array(scaler["input_mean"]), np.array(scaler["input_std"])
        c, d = np.array(scaler["output_mean"]), np.array(scaler["output_std"])
        if is_multivars != -1:
            a, b = a[:,:,is_multivars:is_multivars+1], b[:,:,is_multivars:is_multivars+1]
            c, d = c[:,:,is_multivars:is_multivars+1], d[:,:,is_multivars:is_multivars+1]
        if variable == 'input':
            if scaler_method == 'standard':
                feature = feature * b + a
            else:
                feature = feature * (b-a) + a
        elif variable == 'output':
            if scaler_method == 'standard':
                feature = feature * d + c
            else:
                feature = feature * (d-c) + c
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature
    
    def _get_minmax_scaler(self, X, y, type: str) -> dict:
        scaler = {}
        if type == 'global': 
            scaler["input_mean"] = np.nanmin(X, axis=(0,1), keepdims=True).tolist()
            scaler["input_std"] = np.nanmax(X, axis=(0,1), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmin(y, axis=(0,1), keepdims=True).tolist()
            scaler["output_std"] = np.nanmax(y, axis=(0,1), keepdims=True).tolist()
        elif type == 'region':
            scaler["input_mean"] = np.nanmin(X, axis=(0), keepdims=True).tolist()
            scaler["input_std"] = np.nanmax(X, axis=(0), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmin(y, axis=(0), keepdims=True).tolist()
            scaler["output_std"] = np.nanmax(y, axis=(0), keepdims=True).tolist()
        else:
            raise IOError(f"Unknown variable type {type}")
        return scaler
    
    def _clip_by_date(self, begin_date, end_date, select_begin_date, select_end_date):
        """clip select date on target data"""
        if isinstance(begin_date, int): 
            begin_date = '{year}-01-01'.format(year=begin_date)
        if isinstance(end_date, int): 
            end_date = '{year}-12-31'.format(year=end_date)
        if isinstance(select_begin_date, int): 
            select_begin_date = '{year}-01-01'.format(year=select_begin_date)
        if isinstance(select_end_date, int): 
            select_end_date = '{year}-12-31'.format(year=select_end_date)

        t1 = datetime.datetime.strptime(begin_date,"%Y-%m-%d")
        t2 = datetime.datetime.strptime(end_date,"%Y-%m-%d")
        t3 = datetime.datetime.strptime(select_begin_date,"%Y-%m-%d")
        t4 = datetime.datetime.strptime(select_end_date,"%Y-%m-%d")
        n = (t2-t1).days+1
        begin_idx = (t3-t1).days
        end_idx = (t4-t1).days
        return begin_idx, end_idx
    

