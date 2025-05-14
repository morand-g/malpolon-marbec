import os
import sys
import json
from typing import Callable, Any, Union
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

from sklearn.preprocessing import StandardScaler

# Force work with the malpolon github package localled at the root of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mlpn')))

from malpolon.data.data_module import BaseDataModule

import datetime

SPECTRUM_ALL = ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22', 'qa', 'drad', 'emis', 'emsd', 'trad', 'urad',
                'atran', 'cdist', 'qa_pixel', 'qa_radsat']


class PovertyDataModule(BaseDataModule):
    def __init__(
            self,
            tif_dir: str = '',
            dataset_path: str = '',
            labels_name: str = 'observation_2013+.csv',
            train_batch_size: int = 32,
            inference_batch_size: int = 16,
            num_workers: int = 8,
            fold: str = 'A',
            fold_path: str = 'folds.pkl',
            nature: str = 'composite',
            dict_normalize: str = 'mean_std_normalize_rgb.json',
            **kwargs
    ):

        """DataModule for the Poverty dataset.
        Args:
            tif_dir (str): directory containing the tif files
            dataset_path (str): path to the dataset
            labels_name (str): name of the csv file containing the labels
            train_batch_size (int): batch size for training
            inference_batch_size (int): batch size for inference
            num_workers (int): number of workers for the DataLoader
            fold (int): fold to use for training
            transform (torchvision.transforms): transform to apply to the data"""

        super().__init__()
        dataframe = pd.read_csv(dataset_path + labels_name, sep=";")
        fold_dict = pd.read_pickle(fold_path)
        self.dataframe = dataframe
        self.dataframe_train = dataframe.iloc[fold_dict[fold]['train']]
        self.dataframe_val = dataframe.iloc[fold_dict[fold]['val']]
        self.dataframe_test = dataframe.iloc[fold_dict[fold]['test']]

        self.tif_dir = dataset_path + tif_dir
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.nature = nature
        self.num_workers = num_workers
        self.dict_normalize = json.load(open(dict_normalize, 'r'))

    @property
    def train_transform(self) -> Callable:
        return torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.Normalize(mean=self.dict_normalize["mean"], std=self.dict_normalize["std"]),
        ])

    @property
    def test_transform(self) -> Callable:
        return torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ])

    def get_dataset(self, split: str, transform: Callable, **kwargs) -> Dataset:
        if split == 'train':
            dataset = MSDataset(self.dataframe_train, self.tif_dir, nature=self.nature, transform=transform)
        elif split == 'val':
            dataset = MSDataset(self.dataframe_val, self.tif_dir, nature=self.nature, transform=transform)
        elif split == 'test':
            dataset = MSDataset(self.dataframe_test, self.tif_dir, nature=self.nature, transform=transform)
        elif split == 'all':
            dataset = MSDataset(self.dataframe, self.tif_dir, nature=self.nature, transform=transform)
        return dataset

    def get_all_dataset(self) -> Dataset:
        dataset = MSDataset(self.dataframe, self.tif_dir, nature=self.nature, transform=self.train_transform)
        return dataset

    def train_dataloader(self):
        return DataLoader(self.get_train_dataset(), batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.get_val_dataset(), batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)


    def all_dataloader(self):
        return DataLoader(self.get_all_dataset(), batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)


    def export_predict_csv_basic(self,
                                 predictions: Union[Tensor, np.ndarray],
                                 out_name: str = "predictions",
                                 out_dir: str = './',
                                 return_csv: bool = False,
                                 top_k: int = None,
                                 **kwargs: Any):
        """Export predictions to csv file.

        Exports predictions, probabilities and ids to a csv file.

        Parameters
        ----------
        predictions : Union[Tensor, np.ndarray]
            model's predictions.
        out_name : str, optional
            output CSV file name, by default "predictions"
        out_dir : str, optional
            output directory name, by default "./"
        return_csv : bool, optional
            if true, the method returns the CSV as a pandas DataFrame,
            by default False
        top_k : int, optional
            number of top predictions to return, by default None (max
            number of predictions)

        Returns
        -------
        pandas.DataFrame
            CSV content as a pandas DataFrame if `return_csv` is True
        """
        test_ds = self.get_test_dataset()
        targets = test_ds.targets if test_ds.targets is not None else [-1] * len(predictions)
        df = pd.DataFrame({'ids': test_ds.observation_ids,
                           'predictions': tuple(predictions[:, :top_k].astype(str)),
                           'targets': targets,
                           })
        df['predictions'] = df['predictions'].apply(' '.join)
        df.to_csv(Path(out_dir) / Path(out_name + ".csv"), index=False, sep=',', **kwargs)
        if return_csv:
            return df
        return None


class MSDataset(Dataset):
    """ Dataset returning the LANDSAT tiles and wealth index corresponding to the DHS cluster.
        Rasters were previously downloaded from Earth Engine and stored in the 'landsat_tiles' directory.
        Images contain 8 bands, one of them being a nightlight image. Only the first 7 bands are selected."""

    def __init__(self, dataframe, root_dir, nature="composite", transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.nature = nature
        self.transform = transform
        self.targets = dataframe['iwi'].values
        self.observation_ids = dataframe[['country', 'year', 'cluster_id']].apply(lambda x: '_'.join(x.astype(str)),
                                                                               axis=1).values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        value = row.iwi.astype('float')


        if self.nature == 'composite':

            tile = []
            tile_name = os.path.join(self.root_dir,
                                     str(row.country),
                                     str(row.year),
                                     str(row.cluster_id) + ".tif"
                                     )

            with rasterio.open(tile_name) as src:

                bands = src.descriptions
                bands_indexes = []

                for b in SPECTRUM_ALL:
                    bands_indexes.append(bands.index(b) + 1)

                for band in bands_indexes:
                    layer = src.read(band)
                    tile.append(layer)

            tile = np.stack(tile, axis=0)
            tile = np.nan_to_num(tile)
            tile = self.transform(torch.tensor(tile, dtype=torch.float32))

        elif self.nature == 'seasonal':

            tile = torch.empty((0, 224, 224), dtype=torch.float32)

            for trimester in range(1,5):

                tile_t = []
                tile_name = os.path.join(self.root_dir,
                                         str(row.country),
                                         str(row.year),
                                         str(row.cluster_id) + f"_{trimester}.tif"
                                         )


                with rasterio.open(tile_name) as src:

                    bands = src.descriptions
                    bands_indexes = []

                    for b in SPECTRUM_ALL:
                        bands_indexes.append(bands.index(b) + 1)

                    for band in bands_indexes:
                        layer = src.read(band)
                        tile_t.append(layer)

                tile_t = np.stack(tile_t, axis=0)
                tile_t = np.nan_to_num(tile_t)
                tile_t = self.transform(torch.tensor(tile_t, dtype=torch.float32))
                tile = torch.concat((tile, tile_t), dim=0)
        value = torch.tensor(value, dtype=torch.float32).unsqueeze(-1)

        return tile, value

    def plot(self, idx, rgb=False, save=False):
        """Plot the tile at the given index.
           Args:
                idx (int): index of the tile to plot
                rgb (bool): if True, plot the RGB image, otherwise plot the 19 bands
                save (bool): if True, save the plot in the 'examples/poverty' directory"""

        tile, value = self.__getitem__(idx)

        tile = tile.numpy()

        if rgb:
            fig, ax = pyplot.subplots(1, 1)
            img_rgb = tile[(0, 1, 2), ...].transpose(1, 2, 0)
            ax.imshow(img_rgb)
            ax.set_title(f"Value: {value}, RGB")
        else:

            fig, axs = pyplot.subplots(8, 8)

            for i, ax in enumerate(axs.flat[:]):
                # if i >= len(spectrum):
                #     break
                ax.imshow(tile[i, ...], cmap='pink')

                ax.set_title(f"Band: {SPECTRUM_ALL[i%4]}")

        fig.suptitle(f"Value: {value}")
        if save:
            fig.savefig(f'plot_{idx}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')

        pyplot.tight_layout()
        pyplot.show()
        return tile
