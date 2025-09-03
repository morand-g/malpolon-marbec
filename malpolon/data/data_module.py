"""This module provides a base class for data modules.

Original Authors:   Theo Larcher <theo.larcher@inria.fr>
                    Titouan Lorieul <titouan.lorieul@gmail.com>
Author: Gaetan Morand <gaetan.morand@umontpellier.fr> ;
        Sarah Kiati <sarah.kiati@umontpellier.fr>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import lightning.pytorch as pl

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2

# print("BaseDataModule import torchgeo")

# from torchgeo.datasets import RasterDataset, VectorDataset, random_bbox_assignment, concat_samples, stack_samples
# from torchgeo.datamodules import GeoDataModule
# from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, RandomGeoSampler

# print("Import BaseDataModule torchgeo completed")


if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union

    from torch import Tensor
    from torch.utils.data import Dataset

    import numpy.typing as npt

    Patches = npt.NDArray
    Targets = npt.NDArray



class BaseDataModule(pl.LightningDataModule, ABC):
    """Base class for data modules.

    This class inherits pytorchlightining's LightningDataModule class
    and provides a base class for data modules by re-defining steps
    methods as well as adding new data manipulation methods.
    """
    def __init__(
        self,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers

        # TODO check if uses GPU or not before using pin memory
        self.pin_memory = True

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None
        self.sampler = None
        self.task = None

    @property
    @abstractmethod
    def train_transform(self) -> Callable:
        """Return train data transforms.

        Returns
        -------
        Callable
            train data transforms
        """

    @property
    @abstractmethod
    def test_transform(self) -> Callable:
        """Return test data transforms.

        Returns
        -------
        Callable
            test data transforms
        """

    @abstractmethod
    def get_dataset(self, split: str, transform: Callable, **kwargs: Any) -> Dataset:
        """Return the dataset corresponding to the split.

        Parameters
        ----------
        split : str
            Type of dataset. Values must be on of ["train", "val",
            "test"]
        transform : Callable
            data transforms to apply when loading the dataset

        Returns
        -------
        Dataset
            dataset corresponding to the split
        """

    def get_train_dataset(self) -> Dataset:
        """Call self.get_dataset to return the train dataset.

        Returns
        -------
        Dataset
            train dataset
        """
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform,
        )
        return dataset

    def get_val_dataset(self) -> Dataset:
        """Call self.get_dataset to return the validation dataset.

        Returns
        -------
        Dataset
            validation dataset
        """
        dataset = self.get_dataset(
            split="val",
            transform=self.test_transform,
        )
        return dataset

    def get_test_dataset(self) -> Dataset:
        """Call self.get_dataset to return the test dataset.

        Returns
        -------
        Dataset
            test dataset
        """
        dataset = self.get_dataset(
            split="test",
            transform=self.test_transform,
        )
        return dataset

    # called for every GPU/machine
    def setup(self, stage: Optional[str] = None) -> None:
        """Register the correct datasets to the class attributes.

        Depending on the trainer's stage, this method will retrieve
        the train, val or test dataset and register it as a class
        attribute. The "predict" stage calls for the test dataset.

        Parameters
        ----------
        stage : Optional[str], optional
            trainer's stage, by default None (train)
        """
        if stage in (None, "fit"):
            self.dataset_train = self.get_train_dataset()
            self.dataset_val = self.get_val_dataset()

        if stage == "test":
            self.dataset_test = self.get_test_dataset()

        if stage == "predict":
            self.dataset_predict = self.get_test_dataset()

    def prepare_data(self) -> None:
        """Prepare data.

        Called once on CPU. Class states defined here are lost afterwards.
        This method is intended for data downloading, tokenization,
        permanent transformation...
        """

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            train dataloader
        """
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            Validation dataloader
        """
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            test dataloader
        """
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        """Return predict dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            predict dataloader
        """
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_logits_to_class(self,
                                predictions: Tensor,
                                classes: Union[np.ndarray, Tensor],
                                activation_fn: torch.nn.modules.activation = torch.nn.Softmax(dim=1)) -> Tensor:
        """Convert the model's predictions to class labels.

        This method applies an activation function to the model's
        predictions and returns the corresponding class labels.

        Parameters
        ----------
        predictions : Tensor
            model's predictions (raw logits), by default Softmax(dim=1)
        classes : Union[np.ndarray, Tensor]
            classes labels
        activation_fn : torch.nn.modules.activation, optional
            activation function to apply to the model's predictions,
            by default torch.nn.Softmax(dim=1)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            class labels and corresponding probabilities
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classes = torch.tensor(classes).to(device)
        probas = activation_fn(predictions) if activation_fn is not None else predictions
        if 'binary' in self.task:
            class_preds = probas.round()
        else:
            probas, indices = torch.sort(probas, descending=True)
            probas, indices = probas.to(device), indices.to(device)
            class_preds = torch.zeros_like(probas, device=device)
            for batch_i in range(predictions.shape[0]):  # useful if classes don't span from 0 to n_classes-1
                class_preds[batch_i] = classes[indices[batch_i]]
        return class_preds.to('cpu').numpy().astype(int), probas.to('cpu').numpy()


def load_modality(survey_id, inputs_path, modality):

    if modality == 'sat':
        filename = Path(inputs_path) / "sat" / (survey_id + '.jpg')
        if filename.exists():
            with Image.open(filename) as rgb_patch:
                return v2.functional.pil_to_tensor(rgb_patch).float() / 255
        else:
            return torch.zeros([3, 995, 995]) 
                
    elif modality == 'envhum':
        env = load_modality(survey_id, inputs_path, "env")
        hum = load_modality(survey_id, inputs_path, "hum")
        return torch.from_numpy(np.concatenate([env, hum], axis=0))
        
    elif modality in ('env', 'hum'):
        # 3D cubes
        filename = Path(inputs_path) / modality / (survey_id + '.npy')
        x = np.load(filename).astype(np.float32)
        return torch.from_numpy(np.transpose(x, (2, 0, 1)))

    elif modality == 'dhw':
        # 2D data
        filename = Path(inputs_path) / "dhw" / (survey_id + '.npy')
        x = np.load(filename).astype(np.float32)
        return torch.unsqueeze(torch.from_numpy(x),0)
        
    else:
        # 1D data
        filename = Path(inputs_path) / modality / (survey_id + '.npy')
        x = np.load(filename).astype(np.float32)
        return torch.from_numpy(x)

            


def load_patch(
    survey_id: Union[int, str],
    inputs_path: Path,
    *,
    data: Union[str, list[str]] = "all",
    return_arrays: bool = True,
) -> dict[str, Patches]:
    
    """Load the patch data associated to an observation id.

    Parameters
    ----------
    survey_id : integer / string
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'env', 'sat', 'timeseries'.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : dict containing 3d array-like objects
        Returns a dict containing the requested patches.
    """
    survey_id = str(survey_id)

    patches = {}

    if data == "all":
        data = ['env', 'hum', 'sat', 'timeseries', 'best10']

    for n in data:
        patches[n] = load_modality(survey_id, inputs_path, n)

    return patches


class RLSDataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    region : string, either "both", "fr" or "us"
        Load the observations of both France and US or only a single region.
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    use_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude).
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """
    def __init__(
        self,
        root: Union[str, Path],
        dataset_name: str,
        inputs_path: Union[str, Path],
        subset: str,
        num_classes: int,
        *,
        patch_data: str = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ):
        root = Path(root)

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(f"Possible values for 'subset' are:"
                             f" {possible_subsets} (given {subset})")

        self.root = root
        self.dataset_name = dataset_name
        self.inputs_path = Path(inputs_path)
        self.subset = subset
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform
        self.training = subset != "test"
        self.num_classes = num_classes

        df = self._load_observation_data()

        self.survey_ids = df.index

        first_species_index = list(df.columns).index('eventDate') + 1

        self.species = df.columns[first_species_index:]

        if num_classes == 10:
            self.species = ['Assiculus punctatus', 'Notolabrus parilus', 'Parma mccullochi', 
                            'Coris auricularis', 'Notolabrus gymnogenis', 'Notolabrus tetricus',
                            'Pomacentrus wardi', 'Chrysiptera rollandi', 'Pomacentrus moluccensis',
                            'Halichoeres melanurus']
        
        assert len(self.species) == num_classes

        if self.training:
            self.targets = df[self.species].values.astype(np.float32)

            if self.target_transform:
                self.targets = self.target_transform(self.targets)

            # self.targets = (self.species.values.astype(np.float32) > 0).sum(axis=0)
            assert not (np.isnan(self.targets).any())
        else:
            self.targets = None

    def _load_observation_data(
        self
    ) -> pd.DataFrame:

        df = pd.read_csv(self.inputs_path / self.dataset_name,
                         index_col='survey_id',
                         dtype={22: str, 24: str, 25: str})

        if self.subset == "train+val":
            ind = df.index[df["subset"] == 'train'].union(df.index[df["subset"] == 'val'])
        else:
            ind = df.index[df["subset"] == self.subset]

        df = df.loc[ind].drop(columns='subset')

        return df

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        return len(self.survey_ids)

    def __getitem__(
        self,
        index: int,
    ) -> Union[dict[str, Patches], tuple[dict[str, Patches], Targets]]:
        """Return a dataset item.

        Args:
            index (int): dataset id.

        Returns:
            Union[dict[str, Patches], tuple[dict[str, Patches], Targets]]:
                data and labels corresponding to the dataset id.
        """

        survey_id = self.survey_ids[index]

        patches = load_patch(survey_id, self.inputs_path, data=self.patch_data)

        if self.transform:
            patches = self.transform(patches)

        if self.training:
            target = self.targets[index]

            # if self.target_transform:
            #     target = self.target_transform(target)

            return patches, target
        return patches, -1


class RLSDataModule(BaseDataModule):
    r"""
    Data module for RLS-aus 2024.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        root: str,
        dataset_name: str,
        inputs_path: Union[str, Path],
        num_classes: int,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        target_transform: Callable = None,
        modality_names: Optional[dict[str, str]] = ["env", "hum", "sat"],
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_name = dataset_name
        self.inputs_path = Path(inputs_path)
        self.num_classes = num_classes
        self.root = root
        self.target_transform = target_transform  # check_transform(target_transform)
        self.modality_names = modality_names

    @property
    def train_transform(self):
        return self.general_transform

    @property
    def test_transform(self):
        return self.general_transform

    def general_transform(self, x):

        if 'sat' in x:
            # x['sat'] = v2.functional.center_crop(x['sat'], output_size=384)
            x['sat'] = v2.functional.center_crop(x['sat'], output_size=500)

        return x

    def get_dataset(self, split, transform, **kwargs):

        dataset = RLSDataset(
            self.root,
            self.dataset_name,
            self.inputs_path,
            split,
            self.num_classes,
            patch_data=self.modality_names,
            transform=transform,
            target_transform=self.target_transform,
            **kwargs
        )
        return dataset

    def export_predictions(self,
                           predictions: Union[Tensor, np.ndarray],
                           out_name: str = "predictions",
                           out_dir: str = './',
                           classif: bool = False,
                           probabilities: bool = False,
                           **kwargs: Any):

        #p = Path(out_dir) / Path(out_name + '.csv')
        #print(f"Saving output files to {p}")
        
        test_ds = self.get_test_dataset()

        if classif:
            if probabilities:
                predictions = predictions.softmax(dim=-1)[..., 1]
            else:
                predictions = predictions.argmax(dim=-1)

        df = pd.DataFrame(index=test_ds.survey_ids,
                          columns=test_ds.species,
                          data=predictions)

        df.to_csv(Path(out_dir) / Path(out_name + ".csv"), sep=',', **kwargs)
        #print(df)
        #print("Saved")

        return None

    def export_confusion_matrix(self,
                                dataset_path,
                                predictions: Union[Tensor, np.ndarray],
                                binary: bool = True,
                                out_name: str = "confusion_matrix",
                                out_dir: str = './',
                                **kwargs: Any):

        # Load targets
        df = pd.read_csv(dataset_path, index_col='survey_id',
                         dtype={22: str, 24: str, 25: str})
        first_species_index = list(df.columns).index('eventDate') + 1
        species_columns = df.columns[first_species_index:-1]
        targets = df.loc[df['subset'] == 'test', species_columns]

        if binary:
            targets = (targets != 0).astype(int)

        # Create chart
        numbins = targets.to_numpy().max() + 1

        cm = confusion_matrix(targets.to_numpy().flatten(),
                              predictions.argmax(dim=-1).numpy().flatten(),
                              labels=range(numbins),
                              normalize='true')
        cm = cm.round(2)

        fig, ax = plt.subplots(figsize=(16, 16))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(numbins))
        disp.plot(xticks_rotation='vertical',
                  colorbar=False,
                  text_kw={'fontsize': 'x-small'},
                  ax=ax,
                  **kwargs)
        plt.tight_layout()

        plt.savefig(Path(out_dir) / f"{out_name}.png")

    def get_species_weights(self):

        """ get weights (= number of observations per species) """

        ds = self.get_dataset("train+val", transform=self.train_transform)

        sp_occ = np.log(1+(ds.targets > 0).sum(axis=0)).tolist()

        return sp_occ

    def get_class_weights(self):

        """ get weights (= number of observations per class) """

        ds = self.get_dataset("train+val", transform=self.train_transform)

        class_counts = np.bincount(ds.targets.flatten().astype(int))

        alpha = 1
        weights = 1.0 / (class_counts ** alpha + 1e-6)

        # Normalize to stabilize loss scale
        weights /= weights.sum()

        return weights


# class PopDensGeoDataModule(GeoDataModule):
#     def __init__(self, dataset_class, batch_size, patch_size, length, num_workers, pin_memory, dataset1_path, dataset2_path):
#         super().__init__(dataset_class)
#         self.dataset_class = dataset_class
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.length = length
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.dataset1_path = dataset1_path
#         self.dataset2_path = dataset2_path

#         # initialize raster datasets
#         self.raster_data = RasterDataset(paths=[self.dataset1_path])
#         self.label_data = VectorDataset(paths=[self.dataset2_path], label_name="POPCENSUS22")
#         self.intersect_dataset = self.raster_data & self.label_data  # creating an IntersectionDataset from TorchGeo

#     @property
#     def train_transform(self):
#         transforms = v2.Compose([
#             v2.CenterCrop(512),
#         ])
#         return transforms

#     @property
#     def test_transform(self):
#         transforms = v2.Compose([
#             v2.CenterCrop(512),
#         ])
#         return transforms

#     def get_dataset(self, split, transform, **kwargs):
#         """Return the dataset corresponding to the split.

#                 Parameters
#                 ----------
#                 split : str
#                     Type of dataset. Values must be on of ["train", "val",
#                     "test"]
#                 transform : Callable
#                     data transforms to apply when loading the dataset

#                 Returns
#                 -------
#                 Dataset
#                     dataset corresponding to the split
#         """

#         #define split of dataset
#         generator = torch.Generator().manual_seed(0)
#         (
#             self.train_dataset,
#             self.val_dataset,
#             self.test_dataset,
#         ) = random_bbox_assignment(self.intersect_dataset, [0.6, 0.2, 0.2], generator)
#         if split == "train":
#             dataset = self.train_dataset
#         elif split == "val":
#             dataset = self.val_dataset
#         elif split == "test":
#             dataset = self.test_dataset
#         elif split == "predict":
#             dataset = self.test_dataset
#         else:
#             print("Wrong split partition, valid ones : train, val, test, predict.")

#         return dataset

#     def get_train_dataset(self) -> Dataset:
#         """Call self.get_dataset to return the train dataset.

#         Returns
#         -------
#         Dataset
#             train dataset
#         """
#         dataset = self.get_dataset(
#             split="train",
#             transform=self.train_transform,
#         )

#         return dataset

#     def get_val_dataset(self) -> Dataset:
#         """Call self.get_dataset to return the validation dataset.

#         Returns
#         -------
#         Dataset
#             validation dataset
#         """
#         dataset = self.get_dataset(
#             split="val",
#             transform=self.test_transform,
#         )

#         return dataset

#     def get_test_dataset(self) -> Dataset:
#         """Call self.get_dataset to return the test dataset.

#         Returns
#         -------
#         Dataset
#             test dataset
#         """
#         dataset = self.get_dataset(
#             split="test",
#             transform=self.test_transform,
#         )
#         return dataset

#     def get_predict_dataset(self) -> Dataset:
#         """Call self.get_dataset to return the test dataset.

#         Returns
#         -------
#         Dataset
#             test dataset
#         """
#         dataset = self.get_dataset(
#             split="predict",
#             transform=self.test_transform,
#         )
#         return dataset

#     def setup(self, stage: Optional[str] = None) -> None:
#         """Register the correct datasets to the class attributes.

#                 Depending on the trainer's stage, this method will retrieve
#                 the train, val or test dataset and register it as a class
#                 attribute. The "predict" stage calls for the test dataset.

#                 Parameters
#                 ----------
#                 stage : Optional[str], optional
#                     trainer's stage, by default None (train)
#         """

#         if stage in (None, "fit"):
#             self.dataset_train = self.get_train_dataset()
#             self.train_sampler = RandomGeoSampler(self.dataset_train, size=self.patch_size, length=self.length)

#             self.dataset_val = self.get_val_dataset()
#             self.val_sampler = RandomGeoSampler(self.dataset_val, size=self.patch_size, length=self.length)
#             #GridGeoSampler(self.dataset_val, size=self.patch_size, stride=self.patch_size)

#         if stage == "test":
#             self.dataset_test = self.get_test_dataset()
#             self.test_sampler = GridGeoSampler(self.dataset_test, size=self.patch_size, stride=self.patch_size)

#         if stage == "predict":
#             self.dataset_predict = self.get_test_dataset()
#             self.predict_sampler = GridGeoSampler(self.dataset_predict, size=self.patch_size, stride=self.patch_size)

#     def train_dataloader(self) -> DataLoader:
#         """Return train dataloader instantiated with class attributes.

#         Returns
#         -------
#         DataLoader
#             train dataloader
#         """
#         dataloader = DataLoader(
#             self.dataset_train,
#             sampler=self.train_sampler,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=stack_samples,
#             persistent_workers = True
#         )
#         return dataloader

#     def val_dataloader(self) -> DataLoader:
#         """Return validation dataloader instantiated with class attributes.

#         Returns
#         -------
#         DataLoader
#             Validation dataloader
#         """
#         dataloader = DataLoader(
#             self.dataset_val,
#             sampler=self.val_sampler,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=stack_samples,
#             persistent_workers = True
#         )
#         return dataloader

#     def test_dataloader(self) -> DataLoader:
#         """Return test dataloader instantiated with class attributes.

#         Returns
#         -------
#         DataLoader
#             test dataloader
#         """
#         dataloader = DataLoader(
#             self.dataset_test,
#             sampler=self.test_sampler,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=stack_samples,
#             persistent_workers = True
#         )
#         return dataloader

#     def predict_dataloader(self) -> DataLoader:
#         """Return predict dataloader instantiated with class attributes.

#         Returns
#         -------
#         DataLoader
#             predict dataloader
#         """
#         dataloader = DataLoader(
#             self.dataset_predict,
#             sampler=self.predict_sampler,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             collate_fn=stack_samples
#         )
#         return dataloader