import argparse
import pickle
from pathlib import PosixPath, Path



def get_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', type=str, default='case5')
    parser.add_argument('--model_name', type=str, default='model2')
    parser.add_argument('--test_case', type=str, default='site test')
    parser.add_argument('--testset', type=str, default='train')
    parser.add_argument('--tittle', type=str, default='Train')
    parser.add_argument('--excle_path', type=str, default='/tera07/zhwei/For_QingChen/DataML/plot/xlsx/')
    parser.add_argument('--timeseries_path', type=str, default='/tera07/zhwei/For_QingChen/DataML/plot/timeseries/')
    parser.add_argument('--input_path', type=str, default='/tera07/zhwei/For_QingChen/DataML/FLUXNET/input/')
    parser.add_argument('--output_path', type=str, default='/tera07/zhwei/For_QingChen/DataML/FLUXNET/output/')
    parser.add_argument('--selected', type=str, default='false')

    cfg = vars(parser.parse_args())

    # convert path to PosixPath object
    #cfg["forcing_root"] = Path(cfg["forcing_root"])
    #cfg["et_root"] = Path(cfg["et_root"])
    #cfg["attr_root"] = Path(cfg["attr_root"])
    return cfg


def get_site_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', type=str, default="/tera07/zhwei/For_QingChen/DataML/input/")#input_8D
    parser.add_argument('--outputs_path', type=str, default="/tera07/zhwei/For_QingChen/DataML/output/")#output_8D
    parser.add_argument('--data_path', type=str, default="/tera07/zhwei/For_QingChen/DataML/data/")
    parser.add_argument('--et_product', type=str, default='GLEAM_hybrid')
    parser.add_argument('--begin_year', type=int, default=2003)
    parser.add_argument('--end_year', type=int, default=2019)
    parser.add_argument('--temporal_resolution', type=str, default='1D')
    parser.add_argument('--spatial_resolution', type=str, default='0p25')
    cfg = vars(parser.parse_args())
    return cfg