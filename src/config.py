import argparse
import pickle
from pathlib import PosixPath, Path



def get_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_path', type=str, default="/tera05/lilu/DataML/input/")
    parser.add_argument('--outputs_path', type=str, default="/tera05/lilu/DataML/output/")
    parser.add_argument('--data_path', type=str, default="/tera05/lilu/DataML/data/")
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--et_product', type=str, default='GLEAM_hybrid')
    parser.add_argument('--begin_year', type=int, default=2003)
    parser.add_argument('--end_year', type=int, default=2019)
    parser.add_argument('--et_begin_year', type=int, default=2003)
    parser.add_argument('--et_end_year', type=int, default=2019)
    parser.add_argument('--temporal_resolution', type=str, default='1D')
    parser.add_argument('--spatial_resolution', type=str, default='0p25')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--num_leaves', type=int, default=50)
    parser.add_argument('--n_estimators', type=int, default=100)
    cfg = vars(parser.parse_args())

    # convert path to PosixPath object
    #cfg["forcing_root"] = Path(cfg["forcing_root"])
    #cfg["et_root"] = Path(cfg["et_root"])
    #cfg["attr_root"] = Path(cfg["attr_root"])
    return cfg