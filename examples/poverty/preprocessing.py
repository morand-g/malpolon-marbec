"""Adjacent code for calculating mean and standard deviation of a dataset,
and creating folds for cross-validation.

Author: Isabelle Mornard <isabelle.mornard@umontpellier.fr>"""

from __future__ import annotations

import os
import pickle

import torchvision
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig
import torch
import rasterio

torch.set_float32_matmul_precision('medium')
from poverty_dataset import MSDataModule

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SPECTRUM_ALL = ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22', 'qa', 'drad', 'emis', 'emsd', 'trad', 'urad',
                'atran', 'cdist', 'qa_pixel', 'qa_radsat']

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def calcul_mean_std(cfg: DictConfig) -> None:
    """Calculate the mean and standard deviation of the dataset.
    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """

    datamodule = MSDataModule(**cfg.data, **cfg.task)
    data_loader = datamodule.norm_dataloader()

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

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def check_folds():
    """Check that all items in folds_seasonal.pkl are present in folds.pkl.
    """

    fold = load_pickle("folds.pkl")
    fold1 = load_pickle("folds_seasonal.pkl")

    # Vérification
    all_ok = True
    for key in fold1:  # ex: "A", "B", ...
        if key not in fold:
            print(f"Clé {key} absente dans fold.pkl")
            all_ok = False
            continue

        for subset in fold1[key]:  # ex: "train", "val", "test"
            if subset not in fold[key]:
                print(f"Clé {subset} absente dans fold.pkl[{key}]")
                all_ok = False
                continue

            # Vérifier que chaque item est contenu
            missing = set(fold1[key][subset]) - set(fold[key][subset])
            if missing:
                print(f"Éléments manquants dans fold.pkl[{key}][{subset}] : {missing}")
                all_ok = False

    if all_ok:
        print("✅ Tous les items de fold_1.pkl sont bien présents dans fold.pkl")

def create_tile(tile_name: str, transform) -> torch.Tensor:

    tile = []

    with rasterio.open(tile_name) as src:

        bands = src.descriptions
        bands_indexes = []

        for b in SPECTRUM_ALL:
            bands_indexes.append(bands.index(b) + 1)

        for band in bands_indexes:
            layer = src.read(band)
            tile.append(layer)

    tile_t = np.stack(tile, axis=0)
    tile_t = np.nan_to_num(tile_t)

    return transform(torch.tensor(tile_t, dtype=torch.float32))


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def create_memmap(cfg: DictConfig):
    """Create a memmap file for the dataset.
    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
    """

    file_name = cfg.data.dataset_path + cfg.data.labels_name
    df_csv = pd.read_csv(file_name, sep=";")


    dict_normalize = json.load(open(cfg.data.dict_normalize, 'r'))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(mean=dict_normalize['mean'], std=dict_normalize['std']),
    ])

    memmap_path = "images_memmap.dat"

    X = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(df_csv), 16, 224, 224)
    )

    for i in tqdm(range(len(df_csv))):
        row = df_csv.iloc[i]

        if cfg.data.nature == 'composite':


            tile_name = os.path.join(cfg.data.dataset_path,
                                     str(row.country.lower()),
                                     str(row.year),
                                     str(row.cluster_id) + ".tif"
                                     )

            tile = create_tile(tile_name, transform)

            tile = np.stack(tile, axis=0)
            tile = np.nan_to_num(tile)
            tile = transform(torch.tensor(tile, dtype=torch.float32))

        elif cfg.data.nature == 'seasonal':

            tile = torch.empty((0, 224, 224), dtype=torch.float32)

            for trimester in range(1, 5):

                tile_name = os.path.join(cfg.data.dataset_path,
                                         str(row.country).lower(),
                                         str(row.year),
                                         str(row.cluster_id) + f"_{trimester}.tif"
                                         )

                tile_t = create_tile(tile_name, transform)


                tile = torch.concat((tile, tile_t), dim=0)

        if cfg.data.nightlight:
            tile_name = os.path.join(cfg.data.dataset_path,
                                     f"../HREA/{cfg.data.nightlight}",
                                     str(row.country),
                                     str(row.year),
                                     str(row.cluster_id) + ".tif"
                                     )

            with rasterio.open(tile_name) as src:
                layer = src.read(1)
            tile_n = np.nan_to_num(layer)
            tile_n = transform(torch.tensor(tile_n, dtype=torch.float32).unsqueeze(0))
            tile = torch.concat((tile, tile_n), dim=0)

        X[i] = tile.numpy()

    X.flush()





if __name__ == '__main__':
    df_landsat = pd.read_csv("landsat.csv", sep=";")
    df_landsat = add_rank_column(df_landsat, column_name='fold', seed=42)
    folds = create_folds(df_landsat)

    with open("folds_landsat.pkl", "wb") as f:
        pickle.dump(folds, f)

    print(len(folds['A']['train']), len(folds['A']['val']), len(folds['A']['test']))
    print(len(folds['A']['train'])+len(folds['A']['val'])+len(folds['A']['test']))
