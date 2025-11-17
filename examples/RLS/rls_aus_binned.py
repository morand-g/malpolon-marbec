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



def dictMSE(predictions, targets):

        loss = 0
        
        for mod in targets:
            batch_size, num_patches, _, _ = predictions[mod].shape
            reconstructed_patches = predictions[mod].view(batch_size, num_patches, targets[mod].shape[-2], targets[mod].shape[-1])
            loss += Fmetrics.mean_squared_error( reconstructed_patches,
                                targets[mod])
        
        return loss / len(targets)


class PresenceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int = None,
        num_bins: int = None,
        aggregator: str = 'MLP',
        freeze_submodels: bool = False,
        loss: Union[torch.nn.modules.loss._Loss, str] = "ce_and_sr_loss",
        optimizer: Union[torch.nn.Module, Mapping] = None,
        alpha: Optional[float] = None,
        mae_decoder: bool = False,
        mae_patch_size: int = 4,
        data_sizes: Optional[Mapping] = None,
    ):

        model = MultiModalModel(
            submodels,
            num_species,
            num_bins,
            aggregator,
            freeze_submodels,
            mae_decoder,
            mae_patch_size,
            data_sizes
        )

        # Loss and metrics
        
        if mae_decoder > 0:
            self.metrics = {'mse': dictMSE}
            loss_kwargs = {}
        else:
            self.metrics = {'macro_acc': get_custom_metric(num_bins, 'macro')}
            loss_kwargs = {'num_bins': num_bins,
                   'num_species': num_species,
                   'loss_weights': None}
            
        if loss == "ce_and_sr_loss" and alpha is not None:
            loss_kwargs['alpha'] = alpha


        super().__init__(model, loss, optimizer, loss_kwargs, metrics=self.metrics)

        self.model = model


    def remove_final_layer(self):
        """Remove the final layers of the model to keep only the feature extractor."""

        self.model.aggregator_model[1] = nn.Identity()


    def edit_final_layer(self, new_species_num):
        """Edit the final layer of the model to change the number of output classes."""

        self.model.aggregator_model[1] = nn.Linear(self.model.aggregator_model[1].in_features, new_species_num * self.model.num_bins)


    def pop_last_layers(self):

        avgpool, fc = {}, {}

        for mod in self.model.modality_models:
            avgpool[mod] = self.model.modality_models[mod].avgpool
            fc[mod] = self.model.modality_models[mod].fc

            self.model.modality_models[mod].avgpool = nn.Identity()
            self.model.modality_models[mod].fc = nn.Identity()


            outsize = np.prod(self.data_sizes[mod])
            layerinput = 1024 * int(np.ceil(self.data_sizes[mod][-2] / 32))
            
            self.decoders[mod] = nn.Sequential(
                nn.Linear(layerinput, outsize // 4),
                nn.GELU(),
                nn.Linear(outsize // 4, outsize)
            )
            
        return avgpool, fc
    

    def set_last_layers(self, avgpool, fc):

        for mod in self.model.modality_models:

            self.model.modality_models[mod].avgpool = avgpool[mod]
            self.model.modality_models[mod].fc = fc[mod]
            del(self.model.decoder)


    def _cast_type_to_loss(self, y):

        return {x:y[x].to(torch.float32) for x in y}




@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_fm")
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
    
    reg_system = PresenceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + next(iter(reg_system.metrics)) + "/val:.4f}",
            monitor=next(iter(reg_system.metrics)) + "/val",
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
        

        # Predictions on train+val subset

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

            test_dataset = datamodule.get_test_dataset()
            best_species = list(test_dataset.species)
            save_integrated_gradients(  model_loaded,
                                        test_dataset,
                                        best_species,
                                        class_indices = [list(test_dataset.species).index(s) for s in best_species],
                                        output_dir = Path(cfg.run.checkpoint_path).parent / 'integrated_gradients')

    else:
        if cfg.run.checkpoint_path is not None:

            ##### Change final_layer to be able to load pretrained CP
            #reg_system.edit_final_layer(59)                        # If different number of species
            #avgpool, fc = reg_system.pop_last_layers()             # If using transductive learning
            
                
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])
            
            ##### Rechange final_layer to be able to train
            #reg_system.edit_final_layer(cfg.model.num_species)     # If different number of species
            #reg_system.set_last_layers(avgpool, fc)                # If using transductive learning    

            
        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
