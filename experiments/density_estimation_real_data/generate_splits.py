"""
Script for creating a number of data splits + estimate tails from data
"""

import torch
import numpy as np
import tqdm
from tailnflows.utils import add_raw_data, get_data_path
from tailnflows.models.tail_estimation import estimate_df

def generate_data_split(split, seed, out_path, x):
    torch.manual_seed(seed)

    n = x.shape[0]
    dim = x.shape[1]
    print(
        f'Data: n: {n}, d: {dim}'    
    )
    # get train/val/test split
    print("Split data...")
    n_trn = int(n * 0.4)
    n_val = int(n * 0.2)
    n_tst = n - n_trn - n_val

    trn_ix, val_ix, tst_ix = torch.split(torch.randperm(n), [n_trn, n_val, n_tst])
    
    trn_val_mask = torch.ones(n, dtype=torch.bool)
    trn_val_mask[tst_ix] = False

    # standardize
    print("Standardize data...")
    trn_val_mean = x[trn_val_mask, :].mean(axis=0)
    trn_val_std = x[trn_val_mask, :].std(axis=0)
    x = (x - trn_val_mean) / trn_val_std

    dfs = []
    pos_dfs = []
    neg_dfs = []
    print("Estimate dfs...")
    loop = tqdm.tqdm(range(x.shape[1]))
    
    for dim_ix in loop:
        dim_x = x[trn_val_mask, dim_ix].to("cpu")
        pos_x = dim_x[dim_x > 0].to("cpu")
        neg_x = dim_x[dim_x < 0].to("cpu")

        try:
            dfs.append(estimate_df(dim_x.abs(), verbose=False))
        except ValueError:
            # print(f"ERR: {dim_ix}")
            dfs.append(0)

        try:
            pos_dfs.append(estimate_df(pos_x.abs(), verbose=False))
        except ValueError:
            # print(f"ERR: p {dim_ix}")
            pos_dfs.append(0)

        try:
            neg_dfs.append(estimate_df(neg_x.abs(), verbose=False))
        except ValueError:
            # print(f"ERR: n {dim_ix}")
            neg_dfs.append(0)

    average_df = np.mean(dfs)
    average_pos_df = np.mean(pos_dfs)
    average_neg_df = np.mean(neg_dfs)

    print(f"Average df: {average_df}")
    print(f"Average pos df: {average_pos_df}")
    print(f"Average neg df: {average_neg_df}")

    dataset = {
        "split": {"trn": trn_ix, "val": val_ix, "tst": tst_ix},
        "metadata": {
            "dfs": [float(df) for df in dfs],
            "pos_dfs": [float(df) for df in pos_dfs],
            "neg_dfs": [float(df) for df in neg_dfs],
            "mean": list(trn_val_mean.cpu().numpy()),
            "std": list(trn_val_std.cpu().numpy()),
            "seed": seed,
        },
    }

    split_path = f'{get_data_path()}/{out_path}/{split}'
    print(f'saving to {split_path}...')
    add_raw_data(
        split_path, 
        label="experiment_data", 
        data=dataset, 
        force_write=True
    )


if __name__ == "__main__":
    from functools import partial

    # print('Generating splits + tail estimation for SP500...')
    # from tailnflows.targets.data.sp500_returns import load_return_data
    # # will generate for up to dim top_n_symbols
    # # currently only works for up to dim 145, as the other stocks are incomplete!
    # top_n_symbols = 140
    # x, _ = load_return_data(top_n_symbols)
    # _generator_sp500 = partial(generate_data_split, out_path="splits/sp500", x=x)

    # for split in range(10):
    #     print(f"Preprocessing for SP500 split {split}...")
    #     _generator_sp500(split, seed = 1100 + 100*split)

    # print('Generating splits + tail estimation for fama5...')
    # from tailnflows.targets.data.fama5 import load_data
    # x = load_data()
    # _generator_fama5 = partial(generate_data_split, out_path="splits/fama5", x=x)

    # for split in range(10):
    #     print(f"Preprocessing for fama5 split {split}...")
    #     _generator_fama5(split, seed = 110 + 10*split)

    print('Generating splits + tail estimation for insurance data...')
    from tailnflows.targets.data.insurance import load_data
    x = load_data()
    _generator_insurance = partial(generate_data_split, out_path="splits/insurance", x=x)

    for split in range(10):
        print(f"Preprocessing for insurance data split {split}...")
        _generator_insurance(split, seed = 11 + 1*split)

    print("Process finished successfully.")