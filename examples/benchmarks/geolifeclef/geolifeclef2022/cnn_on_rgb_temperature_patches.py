"""Main script to run training on microlifeclef2022 dataset.

Uses RGB pre-extracted patches and temperature rasters from the dataset.
This script was created for Kaggle participants of the GeoLifeCLEF 2022
challenge.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import PreprocessRGBTemperatureData

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import (GeoLifeCLEF2022Dataset,
                                                    MiniGeoLifeCLEF2022Dataset)
from malpolon.data.environmental_raster import PatchExtractor
from malpolon.logging import Summary
from malpolon.models.custom_models.multi_modal import \
    HomogeneousMultiModalModel
from malpolon.models.standard_prediction_systems import ClassificationSystem


class GeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for GeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        minigeolifeclef: if True, loads MiniGeoLifeCLEF 2022, otherwise loads GeoLifeCLEF2022
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        minigeolifeclef: bool = False,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        task: str = 'classification_multiclass',
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.minigeolifeclef = minigeolifeclef
        self.task = task

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                PreprocessRGBTemperatureData(),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 2,
                    std=[0.229, 0.224, 0.225] * 2,
                ),
                transforms.Lambda(lambda x: {
                    "rgb": x[:3],
                    "temperature": x[3:],
                }),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                PreprocessRGBTemperatureData(),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 2,
                    std=[0.229, 0.224, 0.225] * 2,
                ),
                transforms.Lambda(lambda x: {
                    "rgb": x[:3],
                    "temperature": x[3:],
                }),
            ]
        )

    def get_dataset(self, split, transform, **kwargs):
        if self.minigeolifeclef:
            dataset_cls = MiniGeoLifeCLEF2022Dataset
        else:
            dataset_cls = GeoLifeCLEF2022Dataset

        patch_extractor = PatchExtractor(Path(self.dataset_path) / "rasters", size=20)
        patch_extractor.append("bio_1", nan=-12.0)
        patch_extractor.append("bio_2", nan=1.0)
        patch_extractor.append("bio_7", nan=1.0)

        dataset = dataset_cls(
            self.dataset_path,
            split,
            patch_data=["rgb"],
            use_rasters=True,
            patch_extractor=patch_extractor,
            transform=transform,
            **kwargs
        )
        return dataset


class CustomClassificationSystem(ClassificationSystem):
    def __init__(
        self,
        modalities_model: DictConfig,
        num_outputs: int,
        cfg_optimizer: DictConfig,
        cfg_task: DictConfig,
        checkpoint_path: str = None,
    ):
        model = HomogeneousMultiModalModel(
            ["rgb", "temperature"],
            modalities_model,
            torch.nn.LazyLinear(num_outputs),
        )

        super().__init__(model, **cfg_optimizer, **cfg_task, checkpoint_path=checkpoint_path)


@hydra.main(version_base="1.3", config_path="config", config_name="homogeneous_multi_modal_model")
def main(cfg: DictConfig) -> None:
    # Loggers
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(Path(log_dir)/Path(cfg.loggers.log_dir_name), name=cfg.loggers.exp_name, version="")
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = GeoLifeCLEF2022DataModule(**cfg.data, **cfg.task)
    classif_system = CustomClassificationSystem(**cfg.model, cfg_optimizer=cfg.optim, cfg_task=cfg.task,
                                                checkpoint_path=cfg.run.checkpoint_path)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{top_30_multiclass_accuracy/val:.4f}",
            monitor="top_30_multiclass_accuracy/val",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    if cfg.run.predict:
        model_loaded = CustomClassificationSystem.load_from_checkpoint(classif_system.checkpoint_path,
                                                                       model=classif_system.model,
                                                                       hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(datamodule.get_test_dataset().n_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=3, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])
    else:
        trainer.fit(classif_system, datamodule=datamodule)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()