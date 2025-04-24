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
from captum.attr import IntegratedGradients

from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch import Tensor

import torchmetrics.functional as Fmetrics
import os
import numpy as np
import pandas as pd

OmegaConf.register_new_resolver("eval", eval)

def get_custom_metric(nbins, average_type):

    def custom_metric(predictions, target):

        predictions = predictions.argmax(dim=-1)
        return Fmetrics.classification.multiclass_accuracy(predictions, target, num_classes=nbins, average=average_type)

    return custom_metric


def save_integrated_gradients(model, dataloader, class_indices, output_dir):

    integrated_gradients = IntegratedGradients(model)
    os.makedirs(output_dir, exist_ok=True)
    class_attributions = {class_idx: [] for class_idx in class_indices}

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = {k: v.to(model.device).requires_grad_() for k, v in inputs.items()}
        targets = targets.to(model.device).float().requires_grad_()

        for class_idx in class_indices:
            attributions, delta = integrated_gradients.attribute(inputs, target=targets, return_convergence_delta=True)
            attributions_np = attributions.cpu().numpy()
            class_attributions[class_idx].append({i:attributions_np})

    return(class_attributions)



class PresenceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int,
        num_bins: int,
        freeze_submodels: bool,
        loss: Union[torch.nn.modules.loss._Loss, str] = "modified_ce_loss",
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[dict[str, Callable]] = None,
        loss_kwargs: Optional[Mapping] = {},
    ):

        model = MultiModalModel(
            submodels,
            num_species,
            num_bins,
            freeze_submodels
        )

        metrics = {'micro_acc': get_custom_metric(num_bins, 'micro'),
                   'macro_acc': get_custom_metric(num_bins, 'macro')}

        super().__init__(model, loss, optimizer, loss_kwargs, metrics=metrics)


@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_binned")
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
    datamodule = RLSDataModule(**cfg.data, target_transform=lambda x: (x != 0).astype(float))

    loss_kwargs = {'num_bins': cfg.model.num_bins,
                   'num_species': cfg.model.num_species,
                   'loss_weights': datamodule.get_class_weights()}

    reg_system = PresenceSystem(**cfg.model, **cfg.optim, loss_kwargs=loss_kwargs)

    # Copy current file to log folder
    # copy2(__file__, Path(log_dir) / cfg.run.run_name / Path(__file__).name)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            filename="checkpoint-{epoch:02d}-{step}-{macro_acc/val:.4f}",
            monitor="macro_acc/val",
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
        model_loaded = PresenceSystem.load_from_checkpoint(cfg.run.checkpoint_path)

        # predictions = model_loaded.predict(datamodule, trainer)
        # datamodule.export_predictions(predictions,
        #                               out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        #                               classif=True,
        #                               probabilities=True,
        #                               out_name='predictions-probs')
        # datamodule.export_predictions(predictions,
        #                               out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        #                               classif=True,
        #                               probabilities=False,
        #                               out_name='presences')
        # datamodule.export_confusion_matrix(Path(cfg.data.inputs_path) / cfg.data.dataset_name,
        #                                    predictions,
        #                                    out_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        if cfg.run.interpretable:
            output_dir = Path(cfg.run.checkpoint_path).parent / 'integrated_gradients'

            dataset = datamodule.get_test_dataset()
            species = list(dataset.species)

            best_species = list(pd.read_csv(output_dir.parent / 'best_species.csv', index_col = 0).index)
            class_indices = [species.index(s) for s in best_species]

            test_loader = datamodule.test_dataloader()

            print(f"Dataset length: {len(dataset)}")
            print(f"DataLoader length: {sum(1 for _ in test_loader)}")
            

            atts = save_integrated_gradients(model_loaded, test_loader, class_indices, output_dir)

    else:
        if cfg.run.checkpoint_path is not None:
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])

        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
