"""Main script to run training or inference on Poverty Marbec Dataset.

This script will run the Poverty dataset by default.

Author: Auguste Verdier <auguste.verdier@umontpellier.fr>
        Isabelle Mornard <isabelle.mornard@umontpellier.fr>
"""

from __future__ import annotations

import os
import random

import pandas as pd
import matplotlib.pyplot as plt

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torch import tensor
import torchmetrics.functional as Fmetrics


from poverty_dataset import MSDataModule
from malpolon.logging import Summary
from malpolon.models.standard_prediction_systems import RegressionSystem

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


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

    # Iteration on folds for cross-validation
    for fold in 'ABCDE':
        print("Training fold ", fold)

        # Loggers
        log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        log_dir_fold = os.path.join(log_dir, f"fold_{fold}")
        logger_csv = pl.loggers.CSVLogger(log_dir_fold, name="", version="")
        logger_csv.log_hyperparams(cfg)
        logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=f"tensorboard_logs/fold_{fold}", version="")
        logger_tb.log_hyperparams(cfg)

        # Datamodule & Model
        datamodule = MSDataModule(**cfg.data, fold=fold)
        model = RegressionSystem(cfg.model, **cfg.optim)

        # Lightning Trainer
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
                             **cfg.trainer)

        # Run
        if cfg.run.predict:
            model = RegressionSystem.load_from_checkpoint(cfg.run.checkpoint_path[i],
                                                          model=model.model,
                                                          hparams_preprocess=False,
                                                          weights_dir=log_dir_fold,
                                                          loss=cfg.optim.loss,
                                                          metrics=cfg.optim.metrics)

            # Save prediction points for each fold
            predictions = model.predict(datamodule, trainer)
            np_predictions = predictions.to('cpu').numpy()
            df_predictions = datamodule.export_predict_csv_basic(np_predictions,
                                                                 out_dir=log_dir_fold,
                                                                 out_name=f'predictions_test_dataset_{fold}',
                                                                 return_csv=True)

            inference_data = pd.concat([inference_data, df_predictions])
            i += 1

        else:
            if cfg.run.checkpoint_path:trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path[i])
            else:trainer.fit(model, datamodule=datamodule)
            trainer.validate(model, datamodule=datamodule)
            trainer.test(model, datamodule=datamodule)

    #Gather prediction points over the whole dataset
    if cfg.run.predict:
        inference_data.to_csv(f"{log_dir}/predictions.csv")


@hydra.main(version_base="1.3", config_path="../../../Poverty/config", config_name="cnn_on_ms_torchgeo_config")
def plot_dataset(cfg: DictConfig) -> None:
    """
    Plot a random element of the dataset both in rgb rendering and with whole spectrum.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.

    """
    datamodule = MSDataModule(**cfg.data, **cfg.task)
    dataset = datamodule.get_all_dataset()
    idx = random.randint(0, len(dataset)-1)

    dataset.plot(idx, True)
    dataset.plot(idx, False)


def plot_predict(data: pd.DataFrame):
    """
    Plot prediction points gathered in prediction.csv and compute the global r².

    Parameters
    ----------
    data: pd.DataFrame
    Dataframe containing the prediction points.

    """

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
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Execution time: {minutes} min {seconds:.2f} sec")
