
import os
import json
from typing import Callable, Any, Union
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd

import matplotlib.pyplot as plt
import math

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

from torch.cuda import nvtx

from malpolon.data.data_module import BaseDataModule


SPECTRUM_ALL = ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22', 'lwir']


class MSDataModule(BaseDataModule):
    def __init__(
            self,
            dataset_path: str = '',
            labels_name: str = 'landsat7.csv',
            train_batch_size: int = 32,
            inference_batch_size: int = 16,
            num_workers: int = 8,
            fold: str = 'A',
            fold_path: str = 'folds.pkl',
            nature: str = 'composite',
            nightlight: str = None,
            dict_normalize: str = 'mean_std_normalize_all.json',
            **kwargs
    ):

        """DataModule for the Poverty dataset.
        Args:
            dataset_path (str): path to the dataset
            labels_name (str): name of the csv file containing the labels
            train_batch_size (int): batch size for training
            inference_batch_size (int): batch size for inference
            num_workers (int): number of workers for the DataLoader
            fold (int): fold to use for training
            fold_path (str): path to the file containing the folds
            nature (str): nature of the dataset, either 'composite' or 'seasonal'
            dict_normalize (str): path to the json file containing the mean and std for normalization"""

        super().__init__(train_batch_size, inference_batch_size, num_workers)

        self.dataset_path = dataset_path
        self.labels_name = labels_name
        self.fold_dict = pd.read_pickle(fold_path)[fold]
        self.nature = nature
        self.nightlight = nightlight
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
        return MSDataset(self.dataset_path, self.labels_name,self.fold_dict, split = split, nature=self.nature,
                         nightlight= self.nightlight, transform=transform)

    # def get_dataset(self, split: str, transform: Callable, **kwargs) -> Dataset:
    #     """New approach that reade a np.memmap file"""
    #     print(f"⚡ Loading Fast Memmap Dataset for split: {split}")
    #
    #     # We need to recreate the dataframe logic here because MemmapDataset
    #     # expects a dataframe, but MSDataset used to load it internally.
    #     # We can borrow the helper from the old class or just instantiate it briefly.
    #
    #     # Fast trick: Use the old class just to get the dataframe (it's fast to load CSVs)
    #     # We pass transform=None because we only want the metadata/labels right now
    #     temp_ds = MSDataset(self.dataset_path, self.labels_name, self.fold_dict,
    #                         split=split, nature=self.nature, nightlight=self.nightlight,
    #                         transform=None)

        # return MemmapDataset(
        #     ref_path="poverty_reflectance.dat",
        #     temp_path="poverty_temperature.dat",
        #     meta_path="poverty_meta.npy",
        #     dataframe=temp_ds.dataframe, # Pass the loaded labels
        #     transform=transform
        # )

    def get_all_dataset(self) -> Dataset:
        """Call self.get_dataset to return the whole dataset.

        Returns
        -------
        Dataset
            whole dataset
        """
        dataset = self.get_dataset(
            split="all",
            transform=self.test_transform,
        )
        return dataset

    def get_norm_dataset(self) -> Dataset:
        """Call self.get_dataset to return the whole dataset.

        Returns
        -------
        Dataset
            whole dataset
        """
        dataset = self.get_dataset(
            split="all",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(224),
            ]),
        )
        return dataset

    def train_dataloader(self):
        return DataLoader(self.get_train_dataset(), batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True,prefetch_factor=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.get_val_dataset(), batch_size=self.train_batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,prefetch_factor=4, pin_memory=True)


    def all_dataloader(self):
        return DataLoader(self.get_all_dataset(), batch_size=self.train_batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,prefetch_factor=4, pin_memory=True)

    def norm_dataloader(self):
        return DataLoader(self.get_norm_dataset(), batch_size=self.train_batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,prefetch_factor=4, pin_memory=True)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        nvtx.range_push("transfer_batch_to_device")
        batch_on_device = super().transfer_batch_to_device(batch, device, dataloader_idx)
        nvtx.range_pop()
        return batch_on_device


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


class MemmapDataset(Dataset):
    def __init__(self, ref_path, temp_path, meta_path, dataframe, transform=None):
        self.dataframe = dataframe
        self.observation_ids = dataframe.index
        self.targets = dataframe.iwi.values
        self.transform = transform

        meta = np.load(meta_path, allow_pickle=True).item()

        self.ref_data = np.memmap(ref_path, dtype='uint16', mode='r', shape=meta["shape_ref"])
        self.temp_data = np.memmap(temp_path, dtype='uint16', mode='r', shape=meta["shape_temp"])

        self.ref_offset = meta["ref_offset"]
        self.ref_scale = meta["ref_scale"]
        self.temp_scale = meta["temp_scale"]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        real_idx = self.dataframe.index[idx]
        real_idx = idx

        # Reflectance (T, 6, H, W)
        ref_raw = torch.from_numpy(self.ref_data[real_idx].astype(np.float32))
        ref_img = (ref_raw * self.ref_scale) - self.ref_offset

        # Temperature (T, 1, H, W)
        temp_raw = torch.from_numpy(self.temp_data[real_idx].astype(np.float32))
        temp_img = temp_raw / self.temp_scale

        # Fuse -> (T, 7, H, W)
        full_img = torch.cat([ref_img, temp_img], dim=1)

        # Apply transform PER FRAME
        if self.transform:
            frames = []
            for t in range(full_img.shape[0]):
                frames.append(self.transform(full_img[t]))
            full_img = torch.stack(frames, dim=0)
        full_img = full_img.reshape(4 * 7, 224, 224)

        # Target
        row = self.dataframe.iloc[idx]
        target = torch.tensor(row.iwi, dtype=torch.float32).unsqueeze(-1)

        return full_img, target

    def plot(self, idx, rgb=False):
        """Plot all layers of a given patch.

        A patch is selected based on a key matching the associated
        provider's __get__() method.

        Args:
            item (dict): provider's get index.
        """

        patch, value = self.__getitem__(idx)

        nb_layers = len(SPECTRUM_ALL)

        if rgb:
            patch_rgb = patch[0, [0, 1, 2], :, :]
            img_rgb = patch_rgb.permute(1, 2, 0).numpy()
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())

            # Plot the image
            fig, ax = plt.subplots()
            ax.imshow(img_rgb)
            ax.axis('off')

            plt.suptitle('Tensor for sample: ' + str(idx), fontsize=16)
            plt.show()

        else:
            if nb_layers == 1:
                plt.figure(figsize=(10, 10))
                plt.imshow(patch[0])
            else:
                # calculate the number of rows and columns for the subplots grid
                rows = int(math.ceil(math.sqrt(nb_layers)))
                cols = int(math.ceil(nb_layers / rows))

                # create a figure with a grid of subplots
                fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

                # flatten the subplots array to easily access the subplots
                axs = axs.flatten()

                # loop through the layers of patch data
                for i, band_name in enumerate(SPECTRUM_ALL):
                    # display the layer on the corresponding subplot
                    axs[i].imshow(patch[0][i])
                    axs[i].set_title(f'layer_{i}: {band_name}')
                    axs[i].axis('off')

                # remove empty subplots
                for i in range(nb_layers, rows * cols):
                    fig.delaxes(axs[i])

            plt.suptitle('Tensor for sample: ' + str(idx), fontsize=16)

            # show the plot
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


class MSDataset(Dataset):
    """ Dataset returning the LANDSAT tiles and wealth index corresponding to the DHS cluster.
        Rasters were previously downloaded from Microsoft Planetary Computer.
        Images contain 16 bands, all consigned in the SPECTRUM_ALL variable."""

    def __init__(self, root_dir, labels_name, fold, split,  nature="composite",  nightlight=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            labels_name (string): Path to the csv file with labels
            fold (str): Fold to use for training
            split (str): Split to use for training, validation or testing
            nature (str): Nature of the dataset, either 'composite' or 'seasonal'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        obs_data_columns = {'x': 'lon',
                            'y': 'lat',
                            'IWI': 'iwi'}

        self.dataframe = self._load_observation_data(root_dir, labels_name, split, obs_data_columns, fold)
        self.root_dir = root_dir
        self.nature = nature
        self.nightlight = nightlight
        self.transform = transform


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
                                     str(row.country.lower()),
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

            for trimester in range(1, 5):

                tile_t = []
                tile_name = os.path.join(self.root_dir,
                                         str(row.country).lower(),
                                         str(row.year),
                                         str(row.cluster_id) + f"_{trimester}.tif"
                                         )
                try:

                    with rasterio.open(tile_name) as src:
    
                        bands = src.descriptions
                        bands_indexes = []
    
                        for b in SPECTRUM_ALL:
                            if b=='lwir' and b not in bands:
                                b = 'lwir11'
                            bands_indexes.append(bands.index(b) + 1)
    
                        for band in bands_indexes:
                            layer = src.read(band)
                            tile_t.append(layer)
                            
                except Exception as e:
                    print(f"❌ Erreur avec l'image : {tile_name}, {b}")
                    raise e
                    

                tile_t = np.stack(tile_t, axis=0)
                tile_t = np.nan_to_num(tile_t)
                tile_t = self.transform(torch.tensor(tile_t, dtype=torch.float32))
                tile = torch.concat((tile, tile_t), dim=0)

        if self.nightlight:
            tile_name = os.path.join(self.root_dir,
                                     f"../HREA/{self.nightlight}",
                                     str(row.country),
                                     str(row.year),
                                     str(row.cluster_id) + ".tif"
                                     )

            with rasterio.open(tile_name) as src:
                layer = src.read(1)
            tile_n = np.nan_to_num(layer)
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ])
            tile_n = transforms(torch.tensor(tile_n, dtype=torch.float32).unsqueeze(0))
            tile = torch.concat((tile, tile_n), dim=0)

        value = torch.tensor(value, dtype=torch.float32).unsqueeze(-1)

        return tile, value



    def _load_observation_data(
        self,
        root: str = None,
        obs_fn: str = None,
        subsets: str = ['train', 'test', 'val'],
        keys: dict = {'x': 'lon',
                      'y': 'lat',
                      'IWI': 'iwi'},
        folds: dict = {}
    ) -> pd.DataFrame:
        """Load observation data from a CSV file.

        Reads values from a CSV file containing lon/lat coordinates,
        species id (labels) and dataset subset info (train/test/val).
        The associated columns must have the following values:
        ['longitude', 'latitude', 'speciesId', 'subset']

        If no value is given to root or obs_fn, the method returns an
        empty labels DataFrame.

        Parameters
        ----------
        root : Path
            directory containing the observation (labels) file, by default None.
        obs_fn : str
            observations file name, by default None.
        subsets : str
            desired data subset amongst ["train", "test", "val"], by default
            ["train", "test", "val"] (no restriction).

        Returns
        -------
        pd.DataFrame
            labels DataFrame
        """
        x_key, y_key = keys['x'], keys['y']
        iwi_key = keys['IWI']

        if any([root is None, obs_fn is None]):
            df = pd.DataFrame(columns=[x_key, y_key, iwi_key])
            df = df.iloc[folds[subsets]]
            self.observation_ids = df.index
            self.coordinates = df[["lon", "lat"]].values
            self.targets = df["IWI"].values
            return df
        labels_fp = obs_fn if len(obs_fn.split('.csv')) >= 2 else f'{obs_fn}.csv'
        labels_fp = root + labels_fp
        labels_fp = Path(labels_fp)
        df = pd.read_csv(
            labels_fp,
            sep=";",
        )
        self.unique_labels = np.sort(np.unique(df[iwi_key]))

        if subsets != "all":
            df = df.iloc[folds[subsets]]
        else:
            # Prend tous les indices de tous les splits
            all_indices = []
            for v in folds.values():
                all_indices.extend(v)
            df = df.iloc[all_indices]

        self.observation_ids = df.index
        self.coordinates = df[[x_key, y_key]].values
        self.targets = df[iwi_key].values

        return df


    def plot(self, idx, rgb=False):
        """Plot all layers of a given patch.

        A patch is selected based on a key matching the associated
        provider's __get__() method.

        Args:
            item (dict): provider's get index.
        """

        patch, value = self.__getitem__(idx)

        nb_layers = len(SPECTRUM_ALL)

        if self.nightlight: nb_layers += 1

        if rgb:
            patch_rgb = patch[[0, 1, 2], :, :]
            img_rgb = patch_rgb.permute(1, 2, 0).numpy()
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())

            # Plot the image
            fig, ax = plt.subplots()
            ax.imshow(img_rgb)
            ax.axis('off')

            plt.suptitle('Tensor for sample: ' + str(idx), fontsize=16)
            plt.show()

        else:
            if nb_layers == 1:
                plt.figure(figsize=(10, 10))
                plt.imshow(patch[0])
            else:
                # calculate the number of rows and columns for the subplots grid
                rows = int(math.ceil(math.sqrt(nb_layers)))
                cols = int(math.ceil(nb_layers / rows))

                # create a figure with a grid of subplots
                fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

                # flatten the subplots array to easily access the subplots
                axs = axs.flatten()

                if self.nightlight:
                    SPECTRUM_ALL.append('nightlight')

                # loop through the layers of patch data
                for i, band_name in enumerate(SPECTRUM_ALL):
                    # display the layer on the corresponding subplot
                    axs[i].imshow(patch[i])
                    axs[i].set_title(f'layer_{i}: {band_name}')
                    axs[i].axis('off')

                # remove empty subplots
                for i in range(nb_layers, rows * cols):
                    fig.delaxes(axs[i])

            plt.suptitle('Tensor for sample: ' + str(idx), fontsize=16)

            # show the plot
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

if __name__ == '__main__':
    folds = pd.read_pickle('folds_mada_hrea.pkl')

    print(folds)
