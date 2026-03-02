"""Main script to run training or inference on RLS-aus dataset.

Author: Gaetan Morand <gaetan.morand@umontpellier.fr>
Adapted from: examples/benchmarks/geolifeclef/geolifeclef2022/cnn_on_rgb_temperature_patches.py
"""


from pathlib import Path
from typing import Mapping, Optional, Union

import hydra

from malpolon.data.data_module import RLSDataModule
from malpolon.logging import Summary
from malpolon.models.custom_models import MultiModalModel
from malpolon.models.standard_prediction_systems import GenericPredictionSystem
from malpolon.models.utils import check_metric

from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import torch

import numpy as np

from exports_utils import *

OmegaConf.register_new_resolver("eval", eval)


def dictMSE(predictions, targets):

        loss = 0
        
        for mod in targets:
            loss += Fmetrics.mean_squared_error(predictions[mod].flatten(),
                                                targets[mod].flatten())
        
        return loss / len(targets)
    

class AbundanceSystem(GenericPredictionSystem):
    def __init__(
        self,
        submodels: DictConfig,
        num_species: int,
        aggregator: str = 'MLP',
        freeze_submodels: bool = False,
        loss: Union[torch.nn.modules.loss._Loss, str] = 'filtered_huber_loss',
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[Mapping] = None,
        data_sizes: Optional[Mapping] = None,
        model = None
    ):
        
        if model is None:
            model = MultiModalModel(
                submodels,
                num_species,
                1,
                aggregator,
                freeze_submodels,
                data_sizes=data_sizes
            )

        # Loss and metrics
        
        self.metrics = check_metric(metrics)
        super().__init__(model, loss, optimizer, metrics=self.metrics)

        self.model = model
        self.mae_decoder = False
            


@hydra.main(version_base="1.3", config_path="config", config_name="rls_aus_reg")
def main(cfg: DictConfig) -> None:

    torch.set_float32_matmul_precision('high')


    log_dir = cfg.loggers.log_dir_name
    logger_csv = pl.loggers.CSVLogger(log_dir, name=cfg.run.run_name, version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name=cfg.run.run_name, version="",
                                             default_hp_metric=False)
    logger_tb.log_hyperparams(cfg)

    if "database" in cfg.data.dataset_name:
        # Using species biomass
        target_transform=lambda x: np.log(x+1)
    else:
        # Eco indicators
        target_transform=None
        
    # Datamodule & Model
    datamodule = RLSDataModule(**cfg.data,
                               modality_names= list(cfg.model.submodels.keys()),
                               target_transform=target_transform)
    
    if cfg.run.checkpoint_path is not None:
        
        # Load checkpoint and adapt it if needed
        
        if cfg.run.finetuning_mode == 'transductive':
            # If finetuning from transductive
            reg_system = AbundanceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
            aggregator, avgpool, fc = reg_system.model.pop_last_layers()
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.load_state_dict(checkpoint['state_dict'])
            reg_system.model.set_last_layers(aggregator, avgpool, fc) 
        
        elif cfg.run.finetuning_mode == 'species':
            # If finetuning from different number of species
            reg_system = AbundanceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
            checkpoint = torch.load(cfg.run.checkpoint_path, weights_only=False)
            reg_system.model.edit_final_layer(cp = checkpoint)
            reg_system.load_state_dict(checkpoint['state_dict'])
            reg_system.model.edit_final_layer(new_species_num = cfg.model.num_species)
        
        else:
            # If no change in head (continuing training or inference):
            cp = AbundanceSystem.load_from_checkpoint(cfg.run.checkpoint_path)
            reg_system = AbundanceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes(), model = cp.model)
        
    else:
        reg_system = AbundanceSystem(**cfg.model, **cfg.optim, data_sizes = datamodule.get_data_sizes())
    
                                    
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

        predictions = reg_system.predict(datamodule, trainer)
        
        # Filter present
        
        #presence = pd.read_csv(cfg.run.pa_predictions_path, index_col='survey_id')
        #predictions = predictions.numpy() * presence.to_numpy()

        if cfg.run.testing:
        
            # Predictions on test subset
            datamodule.export_predictions(predictions,
                                      out_dir=Path(cfg.run.checkpoint_path).parent,
                                      out_name='predictions-biomass')
        
        
            export_correlation_scores(cfg)
            
            # Predictions on train+val subset
            cfgtrainval = copy.deepcopy(cfg)
            cfgtrainval.data.dataset_name = cfg.data.dataset_name.split('.')[0] + '_trainval' + '.csv'
            tv_datamodule = RLSDataModule(**cfgtrainval.data,
                                 modality_names= list(cfg.model.submodels.keys()),
                                 target_transform=target_transform)

            tv_predictions = reg_system.predict(tv_datamodule, trainer)
            tv_datamodule.export_predictions(tv_predictions,
                                     out_dir=Path(cfg.run.checkpoint_path).parent,
                                     out_name='predictions-biomass-trainval')
        
        else:
            
            # Predictions on new data set
            
            datamodule.export_predictions(predictions,
                                      out_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
                                      out_name='predictions-biomass')

    else:
        
        trainer.fit(reg_system, datamodule=datamodule)
        trainer.validate(reg_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
