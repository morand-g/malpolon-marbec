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

from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch import nn

import torchmetrics.functional as Fmetrics
import numpy as np
import copy

from exports_utils import *



OmegaConf.register_new_resolver("eval", eval)



def get_custom_metric(nbins, average_type):

    def custom_metric(predictions, target):

        predictions = predictions.argmax(dim=-1)
        return Fmetrics.classification.multiclass_accuracy(predictions, target, num_classes=nbins, average=average_type)

    return custom_metric


class PresenceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int,
        num_bins: int,
        aggregator: str = 'MLP',
        freeze_submodels: bool = False,
        loss: Union[torch.nn.modules.loss._Loss, str] = "ce_and_sr_loss",
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[dict[str, Callable]] = None,
        loss_kwargs: Optional[Mapping] = {},
        alpha: Optional[float] = None,
        mae_decoder: bool = False,
    ):

        model = MultiModalModel(
            submodels,
            num_species,
            num_bins,
            aggregator,
            freeze_submodels,
            mae_decoder
        )

        metrics = {'micro_acc': get_custom_metric(num_bins, 'micro'),
                   'macro_acc': get_custom_metric(num_bins, 'macro')}
        
        if mae_decoder:
            metrics = {'mae_loss': torch.nn.L1Loss()}

        if alpha is not None:
            loss_kwargs['alpha'] = alpha
        
        if mae_decoder:
            metrics = {'mse': Fmetrics.mean_squared_error}
            loss_kwargs = {}

        super().__init__(model, loss, optimizer, loss_kwargs, metrics=metrics)

        self.model = model


    def remove_final_layer(self):
        """Remove the final layers of the model to keep only the feature extractor."""

        self.model.aggregator_model[1] = nn.Identity()


    def edit_final_layer(self, new_species_num):
        """Edit the final layer of the model to change the number of output classes."""

        self.model.aggregator_model[1] = nn.Linear(self.model.aggregator_model[1].in_features, new_species_num * self.model.num_bins)


    def pop_last_layers(self):

        
        avgpool = self.model.modality_models["envhum"].avgpool
        fc = self.model.modality_models["envhum"].fc

        self.model.modality_models["envhum"].avgpool = nn.Identity()
        self.model.modality_models["envhum"].fc = nn.Identity()
        self.model.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 32 * 32 * 19),  # 19 layers, each patch is patch_size x patch_size
        )

        return avgpool, fc



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
    datamodule = RLSDataModule(**cfg.data,
                               modality_names= list(cfg.model.submodels.keys()),
                               target_transform=lambda x: (x != 0).astype(float))
    
    

    loss_kwargs = {'num_bins': cfg.model.num_bins,
                   'num_species': cfg.model.num_species,
                   'loss_weights': None} #datamodule.get_class_weights()}

    reg_system = PresenceSystem(**cfg.model, **cfg.optim, loss_kwargs=loss_kwargs)

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

        # Feature extractor only
        # model_loaded.remove_final_layer()
        # model_loaded.model.classifying = False

        predictions = model_loaded.predict(datamodule, trainer)

        # np.save(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / 'embedding.npy', predictions.numpy())

        # Predictions on test subset
        
        datamodule.export_predictions(predictions,
                                      out_dir=Path(cfg.run.checkpoint_path).parent,
                                      classif=True,
                                      probabilities=True,
                                      out_name='predictions-probs')
        datamodule.export_confusion_matrix(Path(cfg.data.inputs_path) / cfg.data.dataset_name,
                                           predictions,
                                           out_dir=Path(cfg.run.checkpoint_path).parent)
        

        # Predictions on train+val

        cfgtrainval = copy.deepcopy(cfg)
        cfgtrainval.data.dataset_name = cfg.data.dataset_name.split('.')[0] + '_trainval' + '.csv'

        tv_datamodule = RLSDataModule(**cfgtrainval.data,
                               modality_names= list(cfg.model.submodels.keys()),
                               target_transform=lambda x: (x != 0).astype(float))
        
        tv_predictions = model_loaded.predict(tv_datamodule, trainer)

        tv_datamodule.export_predictions(tv_predictions,
                                      out_dir=Path(cfg.run.checkpoint_path).parent,
                                      classif=True,
                                      probabilities=True,
                                      out_name='predictions-probs-trainval')
        
        export_f1_scores(cfg)

        if cfg.run.interpretable:
            output_dir = Path(cfg.run.checkpoint_path).parent / 'integrated_gradients'

            test_dataset = datamodule.get_test_dataset()
            species = list(test_dataset.species)

            best_species = species
            class_indices = [species.index(s) for s in best_species]

            atts = save_integrated_gradients(model_loaded, test_dataset, best_species, class_indices, output_dir)

    else:
        if cfg.run.checkpoint_path is not None:

            ##### Change final_layer to be able to load CP for finetuning on different species
            #reg_system.edit_final_layer(59)
            #avgpool, fc = reg_system.pop_last_layers()
            
                
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])
            
            ##### Rechange final_layer to be able to train
            # reg_system.edit_final_layer(cfg.model.num_species)
            # reg_system.model.modality_models["envhum"].avgpool = avgpool
            # reg_system.model.modality_models["envhum"].fc = fc
            # del(reg_system.model.decoder)

            
        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
