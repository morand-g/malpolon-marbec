"""Main script to run training or inference on Poverty Marbec Dataset.

This script will run the Poverty dataset by default.

Author: Auguste Verdier <auguste.verdier@umontpellier.fr>
        Isabelle Mornard <isabelle.mornard@umontpellier.fr>
"""

from __future__ import annotations

import os
import random

import psutil

import matplotlib
matplotlib.use("Agg")  # backend non interactif pour serveur
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

import pandas as pd
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, BasePredictionWriter

import torch
torch.set_float32_matmul_precision('medium')

from poverty_dataset import MSDataModule
from malpolon.logging import Summary
from malpolon.models.standard_prediction_systems import RegressionSystem


import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# torch.backends.cuda.matmul.allow_tf32 = True # Allow TF32 on CuBlas
# torch.backends.cudnn.allow_tf32 = True       # Allow TF32 on CuDNN

from dali_datamodule import DALIWebDatasetModule

class ResourceMonitor(pl.Callback):
    def _report(self, trainer, phase):
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        swap = psutil.swap_memory()

        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_max = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"\n[RESOURCE] epoch={trainer.current_epoch} phase={phase} "
            f"RAM={memory.rss / 1024**3:.2f} GiB "
            f"VMS={memory.vms / 1024**3:.2f} GiB "
            f"swap_used={swap.used / 1024**3:.2f} GiB "
            f"fds={process.num_fds()} "
            f"GPU_alloc={gpu_allocated:.2f} GiB "
            f"GPU_reserved={gpu_reserved:.2f} GiB "
            f"GPU_max={gpu_max:.2f} GiB"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()
        self._report(trainer, "train_start")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._report(trainer, "val_start")

    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        self._report(trainer, "val_end")


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
    datamodule = DALIWebDatasetModule(
    wds_dir=cfg.data.dataset_path,
    fold=fold,                     # 0-4 for 5-fold CV
    n_folds=5,
    train_batch_size=cfg.data.train_batch_size,
    inference_batch_size=cfg.data.inference_batch_size,
    num_workers=4,              # DALI I/O threads (not PyTorch workers)
)

    datamodule.transfer_batch_to_device = lambda batch, device, idx: batch

    # datamodule = MSDataModule(**cfg.data, fold=fold)
    
    model = RegressionSystem(cfg.model, **cfg.optim)


    # Lightning Trainer
    callbacks = [
        # Summary(),
        ModelCheckpoint(
            dirpath=log_dir_fold,
            filename="best",
            monitor="loss/val",
            mode="min",
            save_on_train_epoch_end=True,
            save_top_k=1,
            save_last=False,
            every_n_train_steps=10,
        ),
        LearningRateMonitor(),
        ResourceMonitor(),
        
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
                                                      metrics=cfg.optim.metrics,
                                                      strict=True,)

        # Save prediction points for each fold
        predictions = model.predict(datamodule, trainer)
        np_predictions = predictions.to('cpu').numpy()
        df_predictions = datamodule.export_predict_csv_basic(np_predictions,
                                                             out_dir=log_dir_fold,
                                                             out_name=f'predictions_test_dataset_{fold}',
                                                             return_csv=True)

        inference_data = pd.concat([inference_data, df_predictions])
        inference_data.to_csv(f"{log_dir}/predictions.csv")

    else:
        if cfg.run.checkpoint_path:
            model = RegressionSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                      model=model.model,
                                                      hparams_preprocess=False,
                                                      weights_dir=log_dir_fold,
                                                      loss=cfg.optim.loss,
                                                      metrics=cfg.optim.metrics,
                                                         strict=True,)
            trainer.fit(model, datamodule=datamodule)
        else:trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def inference(cfg: DictConfig) -> None:

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    model = RegressionSystem(cfg.model, **cfg.optim)
    all_targets = []
    all_predictions = []
    
    for fold in range(5):
        log_dir_fold = os.path.join(log_dir, f"fold_{fold}")
        datamodule = DALIWebDatasetModule(
                        wds_dir=cfg.data.dataset_path,
                        fold=fold,                     # 0-4 for 5-fold CV
                        n_folds=5,
                        train_batch_size=cfg.data.train_batch_size,
                        inference_batch_size=cfg.data.inference_batch_size,
                        num_workers=4,              # DALI I/O threads (not PyTorch workers)
                    )
        
        # Charge le modèle du fold (adapte le chemin)       
        model = RegressionSystem.load_from_checkpoint(os.path.join(log_dir_fold, "best.ckpt"),
                                                      model=model.model,
                                                      hparams_preprocess=False,
                                                      weights_dir=log_dir_fold,
                                                      loss=cfg.optim.loss,
                                                      metrics=cfg.optim.metrics,
                                                      strict=True,)
        model.eval()
        model.to("cuda")
    
        # Récupère les prédictions et cibles pour ce fold
        predictions = []
        targets = []
        for batch in datamodule.test_dataloader():
            tile, label = batch
            tile = tile.cuda()
            label = label.cuda()
            with torch.no_grad():
                pred = model(tile)
            predictions.append(pred.cpu())
            targets.append(label.cpu())
    
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
    
        # Ajoute aux listes globales
        all_targets.append(targets.numpy())
        all_predictions.append(predictions.numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    r2 = r2_score(all_targets, all_predictions)
    df = pd.DataFrame({
        'target': all_targets.flatten(),
        'prediction': all_predictions.flatten()
        })
    
    df.to_csv(f'{log_dir}/all_folds_predictions_vs_targets_mada.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'k--', lw=2)
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title(f'Target vs Prediction (All Folds)\n$R^2 = {r2:.3f}$')
    plt.savefig(f'{log_dir}/all_folds_target_vs_prediction_mada.png', dpi=300, bbox_inches='tight')
    plt.close()



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

    print(f"Plotting sample {idx} from the dataset.")

    dataset.plot(idx, True)
    dataset.plot(idx, False)
    
if __name__ == "__main__":

   main()
