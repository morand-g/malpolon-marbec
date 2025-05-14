"""Main script to run training or inference on Poverty Marbec Dataset.

This script will run the Poverty dataset by default.

Author: Auguste Verdier <auguste.verdier@umontpellier.fr>
        Isabelle Mornard <isabelle.mornard@umontpellier.fr>
"""

from __future__ import annotations

import os
import sys
import random

import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Mapping, Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch import tensor
import torchmetrics.functional as Fmetrics


from poverty_dataset import PovertyDataModule
from malpolon.logging import Summary
from malpolon.models.utils import check_metric, check_model, check_loss
from malpolon.models.standard_prediction_systems import GenericPredictionSystem

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class RegressionSystem(GenericPredictionSystem):
    """Regression task class."""
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        loss: Union[torch.nn.modules.loss._Loss, str],
        optimizer: Union[torch.nn.Module, Mapping] = None,
        lr: float = 1e-2,
        weight_decay: float = 0,
        metrics: Optional[dict[str, Callable]] = None,
        loss_kwargs: Optional[dict] = {},
    ):
        """Class constructor.
        Parameters
        ----------
        model : dict
            model to use
        loss : Union[torch.nn.modules.loss._Loss, str]
            loss or string from the predifined LOSS_CALLABLES.
        optimizer : Union[torch.nn.Module, Mapping]
            optional custom optimizer to use for training
        lr : float
            learning rate
        weight_decay : float
            weight decay
        metrics : dict
            dictionnary containing the metrics to compute.
            Keys must match metrics' names and have a subkey with each
            metric's functional methods as value. This subkey is either
            created from the `malpolon.models.utils.FMETRICS_CALLABLES`
            constant or supplied, by the user directly.
        loss_kwargs: Optional[dict] = {}
            Arguments to be passed to loss constructor.
        """

        metrics = check_metric(metrics)

        self.lr = lr
        self.weight_decay = weight_decay

        model = check_model(model)

        if optimizer is None:
            print(f'[INFO] No optimizer provided: using AdamW with lr={lr}, weight_decay={weight_decay}')
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )


        loss = check_loss(loss)

        super().__init__(model, loss, optimizer, metrics=metrics)


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """

    pl.seed_everything(cfg.seed)

    inference_data = pd.DataFrame([])

    i=0
    for fold in 'ABCDE':
        print("Training fold ", fold)

        log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        log_dir_fold = os.path.join(log_dir, f"fold_{fold}")

        logger_csv = pl.loggers.CSVLogger(log_dir_fold, name="", version="")
        logger_csv.log_hyperparams(cfg)
        logger_tb = pl.loggers.TensorBoardLogger(log_dir_fold, name="tensorboard_logs", version="")
        logger_tb.log_hyperparams(cfg)

        datamodule = PovertyDataModule(**cfg.data, fold=fold)
        model = RegressionSystem(cfg.model, **cfg.optim)

        callbacks = [
            Summary(),
            ModelCheckpoint(
                dirpath=log_dir_fold,
                filename="{epoch:02d}-{step}-{" + f"{next(iter(model.metrics.keys()))}/val" + ":.4f}",
                monitor=f"{next(iter(model.metrics.keys()))}/val",
                mode="max",
                save_on_train_epoch_end=True,
                save_last=True,
                every_n_train_steps=10,
            ),
            LearningRateMonitor()
        ]

        trainer = pl.Trainer(logger=[logger_csv, logger_tb], log_every_n_steps=1, callbacks=callbacks,
                             **cfg.trainer)  #

        if cfg.run.predict:
            model = RegressionSystem.load_from_checkpoint(cfg.run.checkpoint_path[i],
                                                          model=model.model,
                                                          hparams_preprocess=False,
                                                          weights_dir=log_dir_fold,
                                                          loss=cfg.optim.loss,
                                                          metrics=cfg.optim.metrics)

        else:
            if cfg.run.checkpoint_path:trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path[i])
            else:trainer.fit(model, datamodule=datamodule)
            trainer.validate(model, datamodule=datamodule)
            trainer.test(model, datamodule=datamodule)

        predictions = model.predict(datamodule, trainer)
        np_predictions = predictions.to('cpu').numpy()
        df_predictions = datamodule.export_predict_csv_basic(np_predictions,
                                       out_dir=log_dir_fold,
                                       out_name=f'predictions_test_dataset_{fold}',
                                       return_csv=True)

        inference_data = pd.concat([inference_data, df_predictions])
        i+=1

    inference_data.to_csv(f"{log_dir}/predictions.csv")


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def plot_test(cfg: DictConfig, rgb=False) -> None:
    dataM = PovertyDataModule(**cfg.data, **cfg.task)

    dataset = dataM.get_all_dataset()
    idx = random.randint(0, len(dataset)-1)
    dataset.plot(idx, rgb=rgb)


def plot_predict(data: pd.DataFrame):

    # Compute R² score
    r2 = Fmetrics.regression.r2_score(tensor(data['predictions']), tensor(data['targets']), multioutput='uniform_average')

    # Plot predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(data['targets'], data['predictions'], alpha=0.5)
    plt.plot([data['targets'].min(), data['targets'].max()], [data['targets'].min(), data['targets'].max()], 'r--')
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Targets (R² score: {r2:.2f})')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
