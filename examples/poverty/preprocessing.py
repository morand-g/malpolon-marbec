from __future__ import annotations

import os
import pickle
import sys

from tqdm import tqdm
import json
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig
import torch

# Force work with the malpolon github package localled at the root of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlpn')))

torch.set_float32_matmul_precision('medium')
from poverty_dataset import PovertyDataModule

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def calcul_mean_std(cfg: DictConfig) -> None:
    datamodule = PovertyDataModule(**cfg.data, **cfg.task)  #

    datamodule.setup()

    data_loader = datamodule.all_dataloader()

    mean = torch.zeros(16)
    std = torch.zeros(16)

    total_images_count = 0
    for images, _ in tqdm(data_loader):
        batch_images_count = images.size(0)
        images = images.view(batch_images_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count

    mean /= total_images_count
    std /= total_images_count
    print(mean, std)
    with open(cfg.data.dict_normalize, 'w') as f:
        f.write(json.dumps({'mean': mean.tolist(), 'std': std.tolist()}))


def add_rank_column(dataframe: pd.DataFrame, column_name: str = 'fold', seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    num_rows = len(dataframe)
    ranks = ['A', 'B', 'C', 'D', 'E']
    repeated_ranks = ranks * (num_rows // len(ranks)) + ranks[:num_rows % len(ranks)]
    np.random.shuffle(repeated_ranks)
    dataframe[column_name] = repeated_ranks
    return dataframe


def create_folds(dataframe)->dict[str, dict]:
    dataframe = add_rank_column(dataframe)
    fold_names = 'ABCDE'
    test_folds = {i: dataframe[dataframe['fold'] == i].index for i in fold_names}
    folds = {}

    for i, f in enumerate(fold_names):
        folds[f] = {}
        folds[f]['test'] = np.array(test_folds[f])

        val_f = fold_names[(i + 1) % 5]
        folds[f]['val'] = np.array(test_folds[val_f])

        train_fs = [fold_names[(i + 2) % 5], fold_names[(i + 3) % 5], fold_names[(i + 4) % 5]]
        folds[f]['train'] = np.sort(np.concatenate([test_folds[f] for f in train_fs]))

    return folds

if __name__ == '__main__':

    with open("folds.pkl", "rb") as f:
        folds_dict = pickle.load(f)

    df_global = pd.read_csv("C:/Users/Isabelle/Documents/dhs/data/landsat7.csv", sep=";")
    df_mada = pd.read_csv('C:/Users/Isabelle/Documents/dhs/data/madagascar_rural.csv', sep=";")

    folds_mada = {}

    for key1 in folds_dict.keys():
        folds_mada[key1] = {}
        for key2 in folds_dict[key1].keys():
            folds_mada[key1][key2] = []
            for i in folds_dict[key1][key2]:
                row = df_global.iloc[i]
                full_year = 1

                for trimester in range(1, 5):
                    tile_name = os.path.join('C:/Users/Isabelle/Documents/dhs/data/images/seasonal_raw',
                                             str(row.country).lower(),
                                             str(row.year),
                                             str(row.cluster_id) + f"_{trimester}.tif")

                    if not (os.path.exists(tile_name)) or str(row.country).lower() == 'tanzania' or row.urban_rural == 1:
                        full_year = 0

                if full_year:
                    folds_mada[key1][key2].append(i)
            folds_mada[key1][key2] = np.array(folds_mada[key1][key2])

    print(len(folds_mada['A']['test']), len(folds_mada['A']['val']), len(folds_mada['A']['train']))
    print(len(folds_mada['A']['test']) + len(folds_mada['A']['val']) + len(folds_mada['A']['train']))



    with open("folds_mada_rural.pkl", "wb") as f:
        pickle.dump(folds_mada, f)

