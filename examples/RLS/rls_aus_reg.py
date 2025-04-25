"""Main script to run training or inference on RLS-aus dataset.

Author: Gaetan Morand <gaetan.morand@umontpellier.fr>
Adapted from: examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_rgb_temperature_patches.py
"""


from pathlib import Path
from shutil import copy2
from typing import Callable, Mapping, Optional, Union

import hydra

from malpolon.data.data_module import RLSDataModule
from malpolon.logging import Summary
from malpolon.models.custom_models import MultiModalModel
from malpolon.models.standard_prediction_systems import GenericPredictionSystem
from malpolon.models.utils import check_metric

import numpy as np

from omegaconf import DictConfig

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch import Tensor


class AbundanceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int,
        freeze_submodels: bool,
        loss: Union[torch.nn.modules.loss._Loss, str] = 'filtered_huber_loss',
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[dict[str, Callable]] = None,
        loss_weights: Optional[Tensor] = None,
    ):

        model = MultiModalModel(
            submodels,
            num_species,
            -1,
            freeze_submodels
        )

        metrics = check_metric(metrics)

        super().__init__(model, loss, optimizer, metrics=metrics)


@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_reg")
def main(cfg: DictConfig) -> None:

    torch.set_float32_matmul_precision('high')

    # Loggers
    log_dir = cfg.loggers.log_dir_name
    logger_csv = pl.loggers.CSVLogger(log_dir, name=cfg.run.run_name, version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.run.run_name, version="",
                                             default_hp_metric=False)
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = RLSDataModule(**cfg.data, target_transform=lambda x: np.log(x+1))
    reg_system = AbundanceSystem(**cfg.model, **cfg.optim)

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
        model_loaded = AbundanceSystem.load_from_checkpoint(cfg.run.checkpoint_path)

        predictions = model_loaded.predict(datamodule, trainer)

        # Load predicted_presence
        presence = pd.read_csv(cfg.run.pa_predictions_path, index_col='survey_id')
        predictions = predictions.numpy() * presence.to_numpy()

        datamodule.export_predictions(predictions,
                                      out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    else:
        if cfg.run.checkpoint_path is not None:
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])

        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
