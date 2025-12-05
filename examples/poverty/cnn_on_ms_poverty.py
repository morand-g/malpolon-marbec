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

from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands


import torch
import torch.nn as nn
from torch import tensor
import torchmetrics.functional as Fmetrics

# torch.set_float32_matmul_precision('medium')


from poverty_dataset import MSDataModule
from malpolon.logging import Summary
from malpolon.models.standard_prediction_systems import RegressionSystem


import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

torch.backends.cuda.matmul.allow_tf32 = True # Allow TF32 on CuBlas
torch.backends.cudnn.allow_tf32 = True       # Allow TF32 on CuDNN


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

    fold = cfg.run.fold

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

    factory = EncoderDecoderFactory()
    base_model = factory.build_model(
        task="classification",
        backbone="prithvi_eo_v2_300",
        backbone_bands=[
            HLSBands.RED,
            HLSBands.GREEN,
            HLSBands.BLUE,
            HLSBands.NIR_NARROW,
            HLSBands.SWIR_1,
            HLSBands.SWIR_2,
        ],
        backbone_freeze_backbone=True,
        backbone_pretrained=True,
        decoder="FCNDecoder",
        num_classes=1,
        backbone_in_channels=6,  # correspond aux 6 canaux des données
    )
    base_model = TerraTorchWrapper(base_model)

    # print params number of the model trainable and non trainable
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    model = RegressionSystem(base_model, **cfg.optim)

    # model = RegressionSystem(cfg.model, **cfg.optim)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir_fold,
            filename="{epoch:02d}-{step}-{" + f"{next(iter(model.metrics.keys()))}_val" + ":.4f}",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=10,
        ),
        LearningRateMonitor()
    ]

    trainer = pl.Trainer(logger=[logger_csv, logger_tb], log_every_n_steps=1, callbacks=callbacks,
                         **cfg.trainer)

    if os.path.exists(f"{log_dir}/predictions.csv"):
        inference_data = pd.read_csv(f"{log_dir}/predictions.csv", index_col=0)
    else: inference_data = pd.DataFrame([])

    # Run
    if cfg.run.predict:
        model = RegressionSystem.load_from_checkpoint(cfg.run.checkpoint_path,
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

    else:
        if cfg.run.checkpoint_path:trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        else:trainer.fit(model, datamodule=datamodule)
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

    #Gather prediction points over the whole dataset
    if cfg.run.predict:
        inference_data.to_csv(f"{log_dir}/predictions.csv")

class TerraTorchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # TerraTorch ModelOutput → dictionnaire ou objet avec attribut .output
        if isinstance(out, dict):
            return out["output"]
        else:
            return out.output


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
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


if __name__ == "__main__":
    main()
