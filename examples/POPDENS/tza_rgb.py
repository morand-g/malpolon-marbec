"""Main script to run training or inference on TZA_PopDensity dataset.

Adapted from: examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_rgb_temperature_patches.py
"""


from pathlib import Path
from shutil import copy2
from typing import Callable, Mapping, Optional, Union

import hydra

from malpolon.logging import Summary
from malpolon.data.data_module import PopDensGeoDataModule
from malpolon.models.standard_prediction_systems import RegressionSystem
from malpolon.models.utils import check_metric

import numpy as np
import pandas as pd
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchgeo.datasets import RasterDataset
from torchgeo.datamodules import GeoDataModule
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler



@hydra.main(version_base="1.3", config_path="config", config_name="cnn_rgb_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    #torch.set_float32_matmul_precision('high') # flag for internal precision of float32 matrix multiplications

    # Loggers
    log_dir = cfg.loggers.log_dir_name
    logger_csv = pl.loggers.CSVLogger(log_dir, name=cfg.run.run_name, version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.run.run_name, version="",
                                             default_hp_metric=False)
    logger_tb.log_hyperparams(cfg)

    # initialize raster datasets
    raster_data = RasterDataset(paths = [cfg.data.inputs_path])
    label_data = RasterDataset(paths=[cfg.data.labels_path])
    dataset = raster_data & label_data  # creating an IntersectionDataset from TorchGeo

    # Datamodule & Model
    datamodule = PopDensGeoDataModule(
        dataset_class=type(dataset),  # intersection dataset class
        batch_size=cfg.data.batch_size,
        patch_size=cfg.data.patch_size,
        length=cfg.data.sample_size,
        num_workers=cfg.data.num_workers,
        dataset1=raster_data,  # passed to the Intersection dataset init
        dataset2=label_data,  # passed to the Intersection dataset init
    )
    reg_system = RegressionSystem(cfg.model, cfg.optim)

    # Copy current file to log folder
    # copy2(__file__, Path(log_dir) / cfg.run.run_name / Path(__file__).name)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            filename="checkpoint-{epoch:02d}-{step}-{r2/val:.4f}",
            monitor="r2/val",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    # Training / Inference

    if cfg.run.predict:
        model_loaded = RegressionSystem.load_from_checkpoint(cfg.run.checkpoint_path)

        predictions = model_loaded.predict(datamodule, trainer)

        # Load predicted_presence
        datamodule.export_predictions(predictions.numpy(),
                                      out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    else:
        if cfg.run.checkpoint_path is not None:
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])

        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
