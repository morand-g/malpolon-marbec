"""Adjacent code for calculating mean and standard deviation of a dataset,
and creating folds for cross-validation.

Author: Isabelle Mornard <isabelle.mornard@umontpellier.fr>"""

from __future__ import annotations

import os
import pickle

from tqdm import tqdm
import json
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig
import torch

torch.set_float32_matmul_precision('medium')
from poverty_dataset import MSDataModule

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

@hydra.main(version_base="1.3", config_path="../../../Poverty/config", config_name="cnn_on_ms_torchgeo_config")
def calcul_mean_std(cfg: DictConfig) -> None:
    """Calculate the mean and standard deviation of the dataset.
    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """

    datamodule = MSDataModule(**cfg.data, **cfg.task)
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
    """Add a column with ranks A, B, C, D, E to the dataframe.
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to which the rank column will be added.
    column_name : str, optional
        The name of the column to be added, by default 'fold'.
    seed : int, optional
        The seed for random number generation, by default 42.
    Returns
    -------
    pd.DataFrame
        The dataframe with the added rank column.
    """

    np.random.seed(seed)
    num_rows = len(dataframe)
    ranks = ['A', 'B', 'C', 'D', 'E']
    repeated_ranks = ranks * (num_rows // len(ranks)) + ranks[:num_rows % len(ranks)]
    np.random.shuffle(repeated_ranks)
    dataframe[column_name] = repeated_ranks
    return dataframe


def create_folds(dataframe)->dict[str, dict]:
    """Create folds for cross-validation from the dataframe.
    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the data to be split into folds.
    Returns
    -------
    dict[str, dict]
        A dictionary containing the folds, where each key is a fold name (A, B, C, D, E)
        and each value is another dictionary
        with keys 'test', 'val', and 'train' containing the indices of the respective folds.
    """

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

    df_global = pd.read_csv("../../../images/landsat7.csv", sep=";")

    folds_season = {}

    for key1 in folds_dict.keys():
        folds_season[key1] = {}
        print(key1)
        for key2 in folds_dict[key1].keys():
            folds_season[key1][key2] = []
            print(key2)
            for i in folds_dict[key1][key2]:
                row = df_global.iloc[i]
                full_year = 1

                print(i)

                for trimester in range(1, 5):
                    tile_name = os.path.join('../../../images/seasonal_raw',
                                             str(row.country).lower(),
                                             str(row.year),
                                             str(row.cluster_id) + f"_{trimester}.tif")

                    if not (os.path.exists(tile_name)):
                        full_year = 0

                if full_year:
                    folds_season[key1][key2].append(i)
            folds_season[key1][key2] = np.array(folds_season[key1][key2])

    print(len(folds_season['A']['test']), len(folds_season['A']['val']), len(folds_season['A']['train']))
    print(len(folds_season['A']['test']) + len(folds_season['A']['val']) + len(folds_season['A']['train']))



    with open("folds_seasonal.pkl", "wb") as f:
        pickle.dump(folds_season, f)
