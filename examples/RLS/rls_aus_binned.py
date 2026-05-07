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

from deep_utils import *



OmegaConf.register_new_resolver("eval", eval)



def get_custom_metric(nbins, average_type):

    def custom_metric(predictions, target):

        predictions = predictions.argmax(dim=-1)
        return Fmetrics.classification.multiclass_accuracy(predictions, target, num_classes=nbins, average=average_type)

    return custom_metric



def dictMSE(predictions, targets):

        loss = 0
        
        for mod in targets:
            loss += Fmetrics.mean_squared_error(predictions[mod].flatten(),
                                                targets[mod].flatten())
        
        return loss / len(targets)


class PresenceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int = None,
        num_bins: int = None,
        aggregator: str = 'MLP',
        freeze_submodels: bool = False,
        loss: Union[torch.nn.modules.loss._Loss, str] = "mae_loss",
        optimizer: Union[torch.nn.Module, Mapping] = None,
        alpha: Optional[float] = None,
        mae_decoder: bool = False,
        mae_patch_size: int = 8,
        data_sizes: Optional[Mapping] = None,
        model = None,
        mse_alpha = 0
    ):
        
        if model is None:
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
        
        loss_kwargs = {}
        
        if mae_decoder > 0:
            self.metrics = {'mse': dictMSE}
            
        else:
            self.metrics = {'macro_acc': get_custom_metric(num_bins, 'macro')}
            
            if loss == 'modified_ce_loss':
                loss_kwargs = {'num_bins': num_bins,
                       'num_species': num_species,
                       'loss_weights': None}
                if mse_alpha > 0:
                    loss_kwargs['mse_alpha'] = mse_alpha
            
            elif loss == "ce_and_sr_loss" and alpha is not None:
                loss_kwargs['alpha'] = alpha
                

        super().__init__(model, loss, optimizer, loss_kwargs, metrics=self.metrics)

        self.model = model
        self.mae_decoder = mae_decoder
        self.num_bins = num_bins
            


    def _cast_type_to_loss(self, y):
        
        if self.mae_decoder:
            return {x:y[x].to(torch.float32) for x in y}
        else:
            return super()._cast_type_to_loss(y)




@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_binned_pa")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.run.seed, workers=True)

    torch.set_float32_matmul_precision('high')


    log_dir = cfg.loggers.log_dir_name
    logger_csv = pl.loggers.CSVLogger(log_dir, name=cfg.run.run_name, version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.run.run_name, version="",
                                             default_hp_metric=False)
    logger_tb.log_hyperparams(cfg)

    
    # Define target transform
    
    if cfg.model.num_bins == 2:
        target_transform=lambda x: (x != 0).astype(float)
    else:
        target_transform=lambda x: x.astype(int)
        
    # Datamodule & Model
    datamodule = RLSDataModule(**cfg.data,
                               modality_names= list(cfg.model.submodels.keys()),
                               target_transform=target_transform)
    
    if cfg.run.checkpoint_path is not None:
        
        # Load checkpoint and adapt it if needed
        
        if cfg.run.finetuning_mode == 'transductive':
            # If finetuning from transductive
            reg_system = PresenceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
            aggregator, avgpool, fc = reg_system.model.pop_last_layers()
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])
            reg_system.model.set_last_layers(aggregator, avgpool, fc) 
        
        elif cfg.run.finetuning_mode == 'species':
            # If finetuning from different number of species
            reg_system = PresenceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.model.edit_final_layer(cp = checkpoint)
            reg_system.load_state_dict(checkpoint['state_dict'])
            reg_system.model.edit_final_layer(new_species_num = cfg.model.num_species)
        
        else:
            # If no change in head (continuing training or inference):
            cp = PresenceSystem.load_from_checkpoint(cfg.run.checkpoint_path, weights_only=False)
            reg_system = PresenceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes(), model = cp.model)
        
    else:
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

    if cfg.run.inference:


        if cfg.run.embeddings:
            # Feature extractor only

            reg_system.model.remove_final_layer()
            reg_system.model.classifying = False

            predictions = reg_system.predict(datamodule, trainer)

            np.save(Path(cfg.run.checkpoint_path).parent / 'embedding.npy', predictions.numpy())

            cfgtrainval = copy.deepcopy(cfg)
            cfgtrainval.data.dataset_name = cfg.data.dataset_name.split('.')[0] + '_trainval' + '.csv'

            tv_datamodule = RLSDataModule(**cfgtrainval.data,
                                    modality_names= list(cfg.model.submodels.keys()),
                                    target_transform=lambda x: (x != 0).astype(float))
            
            tv_predictions = reg_system.predict(tv_datamodule, trainer)

            np.save(Path(cfg.run.checkpoint_path).parent / 'embedding_trainval.npy', tv_predictions.numpy())

        elif cfg.run.interpretable:

            test_dataset = datamodule.get_test_dataset()
            best_species = list(test_dataset.species)
            save_integrated_gradients(  reg_system, test_dataset, best_species,
                                        class_indices = [list(test_dataset.species).index(s) for s in best_species],
                                        output_dir = Path(cfg.run.checkpoint_path).parent / 'integrated_gradients')

        else:
            
            predictions = reg_system.predict(datamodule, trainer)

            if cfg.run.testing:
                
                # Predictions on test subset
                datamodule.export_predictions(predictions,
                                                out_dir=Path(cfg.run.checkpoint_path).parent,
                                                classif=True, out_name='predictions-probs')
                ## Binary
                # Predictions on train+val subset
                cfgtrainval = copy.deepcopy(cfg)
                cfgtrainval.data.dataset_name = cfg.data.dataset_name.split('.')[0] + '_trainval' + '.csv'
                tv_datamodule = RLSDataModule(**cfgtrainval.data,
                                     modality_names= list(cfg.model.submodels.keys()),
                                     target_transform=lambda x: (x != 0).astype(float))
           
                tv_predictions = reg_system.predict(tv_datamodule, trainer)
                tv_datamodule.export_predictions(tv_predictions,
                                         out_dir=Path(cfg.run.checkpoint_path).parent,
                                         classif=True, out_name='predictions-probs-trainval')
           
                if cfg.model.num_bins == 2:
                    export_f1_scores(cfg)
                    
                    score_path = next(Path(cfg.run.checkpoint_path).parent.glob("testF1*.csv")).name
                    threshold = float(score_path.split("TH=")[1].split("--")[0])

                    datamodule.export_predictions((predictions[...,1] >= threshold).int(),
                                                    out_dir=Path(cfg.run.checkpoint_path).parent,
                                                    classif=False, out_name=f"presences--TH={threshold}")
                else:
                    export_correlation_scores(cfg, classif = True)
                    
            else:
                
                # Predictions on new data set
                
                datamodule.export_predictions(predictions,
                                                out_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
                                                classif=True, out_name='predictions-probs')
                
                score_path = next(Path(cfg.run.checkpoint_path).parent.glob("testF1*.csv")).name
                threshold = float(score_path.split("TH=")[1].split("--")[0])
                
                datamodule.export_predictions((predictions[...,1] >= threshold).int(),
                                                out_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
                                                classif=False, out_name=f"presences--TH={threshold}")

        
    else:
        
        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
