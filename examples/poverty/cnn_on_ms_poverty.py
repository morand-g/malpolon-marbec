"""Main script to run training or inference on Poverty Marbec Dataset.

This script will run the Poverty dataset by default.

Author: Auguste Verdier <auguste.verdier@umontpellier.fr>
        Isabelle Mornard <isabelle.mornard@umontpellier.fr>
"""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Optional

import hydra
import lightning.pytorch as pl
import matplotlib
import numpy as np
import optuna
import pandas as pd
import psutil
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback

from rasterio.errors import NotGeoreferencedWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dali_datamodule import DALIWebDatasetModule
from poverty_dataset import MSDataModule
from malpolon.models.standard_prediction_systems import RegressionSystem

import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

torch.set_float32_matmul_precision("medium")

class ResourceMonitor(pl.Callback):
    def _report(self, trainer: pl.Trainer, phase: str) -> None:
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        swap = psutil.swap_memory()

        values = {"gpu_alloc": 0.0,
                  "gpu_reserved": 0.0,
                  "gpu_max": 0.0,
                 }

        if torch.cuda.is_available():
            values["gpu_alloc"] = (
                torch.cuda.memory_allocated() / 1024**3
            )
            values["gpu_reserved"] = (
                torch.cuda.memory_reserved() / 1024**3
            )
            values["gpu_max"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )

        print(
            f"\n[RESOURCE] epoch={trainer.current_epoch} "
            f"phase={phase} "
            f"RAM={memory.rss / 1024**3:.2f} GiB "
            f"VMS={memory.vms / 1024**3:.2f} GiB "
            f"swap_used={swap.used / 1024**3:.2f} GiB "
            f"fds={process.num_fds()} "
            f"GPU_alloc={values['gpu_alloc']:.2f} GiB "
            f"GPU_reserved={values['gpu_reserved']:.2f} GiB "
            f"GPU_max={values['gpu_max']:.2f} GiB"
        )

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._report(trainer, "train_start")

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._report(trainer, "val_start")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._report(trainer, "val_end")

class OptunaPruningCallback(pl.Callback):
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # Ignore la validation de sanity check lancée avant l'entraînement.
        if trainer.sanity_checking:
            return

        metric = trainer.callback_metrics.get(self.monitor)

        if metric is None:
            return

        value = float(metric.detach().cpu())
        step = trainer.current_epoch

        self.trial.report(value, step=step)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {step}: "
                f"{self.monitor}={value:.6f}"
            )

def create_datamodule(
    cfg: DictConfig,
    fold: int,
    train_batch_size: Optional[int] = None,
) -> DALIWebDatasetModule:
    datamodule = DALIWebDatasetModule(
        wds_dir=cfg.data.dataset_path,
        fold=fold,
        n_folds=5,
        train_batch_size=(
            train_batch_size
            if train_batch_size is not None
            else cfg.data.train_batch_size
        ),
        inference_batch_size=cfg.data.inference_batch_size,
        num_workers=cfg.data.get("num_workers", 4),
    )

    datamodule.transfer_batch_to_device = (
        lambda batch, device, dataloader_idx: batch
    )

    return datamodule

def create_loggers(
    cfg: DictConfig,
    output_dir: Path,
    fold: int,
    trial_number: Optional[int] = None,
):
    if trial_number is None:
        run_dir = output_dir / f"fold_{fold}"
    else:
        run_dir = (
            output_dir
            / "optuna"
            / f"trial_{trial_number:04d}"
            / f"fold_{fold}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    logger_csv = pl.loggers.CSVLogger(
        save_dir=str(run_dir),
        name="",
        version="",
    )

    logger_tb = pl.loggers.TensorBoardLogger(
        save_dir=str(output_dir),
        name=(
            f"tensorboard_logs/fold_{fold}"
            if trial_number is None
            else f"tensorboard_logs/"
                 f"trial_{trial_number:04d}/fold_{fold}"
        ),
        version="",
    )

    hyperparameters = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )

    logger_csv.log_hyperparams(hyperparameters)
    logger_tb.log_hyperparams(hyperparameters)

    return run_dir, [logger_csv, logger_tb]

def create_callbacks(
    run_dir: Path,
    trial: Optional[optuna.Trial] = None,
    enable_resource_monitor: bool = True,
):
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        every_n_epochs=1,
    )
    
    patience = 20 if trial is not None else 30
    
    early_stopping = EarlyStopping(
        monitor="loss/val",
        mode="min",
        patience=patience,
        min_delta=0.0,
        check_finite=True,
        verbose=True,
    )

    callbacks = [
        checkpoint,
        early_stopping,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if enable_resource_monitor:
        callbacks.append(ResourceMonitor())

    pruning_callback = None

    if trial is not None:
        pruning_callback = OptunaPruningCallback(
        trial=trial,
        monitor="loss/val",
    )
        callbacks.append(pruning_callback)

    return callbacks, checkpoint, pruning_callback

def train_fold(
    cfg: DictConfig,
    fold: int,
    trial: Optional[optuna.Trial] = None,
) -> float:
    pl.seed_everything(cfg.seed, workers=True)

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    trial_number = None if trial is None else trial.number

    run_dir, loggers = create_loggers(
        cfg=cfg,
        output_dir=output_dir,
        fold=fold,
        trial_number=trial_number,
    )

    batch_size = cfg.data.train_batch_size

    if trial is not None:
        batch_size = trial.suggest_categorical(
            "train_batch_size",
            [16, 32, 64],
        )

    datamodule = create_datamodule(
        cfg=cfg,
        fold=fold,
        train_batch_size=batch_size,
    )

    model = RegressionSystem(
        cfg.model,
        **cfg.optim,
    )

    callbacks, checkpoint, pruning_callback = create_callbacks(
        run_dir=run_dir,
        trial=trial,
        enable_resource_monitor=trial is None,
    )

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        **cfg.trainer,
    )

    try:
        if cfg.run.checkpoint_path and trial is None:
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=cfg.run.checkpoint_path,
            )
        else:
            trainer.fit(
                model=model,
                datamodule=datamodule,
            )


        best_score = checkpoint.best_model_score

        if best_score is None:
            available_metrics = list(
                trainer.callback_metrics.keys()
            )
            raise RuntimeError(
                "'loss/val' n'a pas été trouvée par ModelCheckpoint. "
                f"Métriques disponibles : {available_metrics}"
            )

        best_mse = float(best_score.detach().cpu())

        if trial is None:
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path="best",
            )

        return best_mse

    finally:
        del trainer
        del model
        del datamodule

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

def objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
) -> float:
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            base_cfg,
            resolve=True,
        )
    )

    cfg.optim.optimizer.adamw.kwargs.lr = trial.suggest_float(
        "lr",
        1e-6,
        3e-4,
        log=True,
    )

    cfg.optim.optimizer.adamw.kwargs.weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    return train_fold(
        cfg=cfg,
        fold=cfg.run.fold,
        trial=trial,
    )

def run_hpo(cfg: DictConfig) -> None:
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    storage_path = output_dir / "optuna.db"

    sampler = optuna.samplers.TPESampler(
        seed=cfg.seed,
        n_startup_trials=5,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=40,
        interval_steps=5,
        n_min_trials=3,
    )

    study = optuna.create_study(
        study_name="resnet18_regression",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.hpo.n_trials,
        gc_after_trial=True,
    )

    print(f"Best validation MSE: {study.best_value}")
    print("Best hyperparameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    best_params_path = output_dir / "best_optuna_params.yaml"

    OmegaConf.save(
        config=OmegaConf.create(study.best_params),
        f=best_params_path,
    )

def predict_fold(
    cfg: DictConfig,
    fold: int,
) -> Path:
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    fold_dir = output_dir / f"fold_{fold}"

    checkpoint_path = (
        Path(cfg.run.checkpoint_path)
        if cfg.run.checkpoint_path
        else fold_dir / "best.ckpt"
    )

    datamodule = create_datamodule(cfg, fold)

    base_model = RegressionSystem(
        cfg.model,
        **cfg.optim,
    )

    model = RegressionSystem.load_from_checkpoint(
        str(checkpoint_path),
        model=base_model.model,
        hparams_preprocess=False,
        weights_dir=str(fold_dir),
        loss=cfg.optim.loss,
        metrics=cfg.optim.metrics,
        strict=True,
    )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", 1),
        logger=False,
        enable_checkpointing=False,
    )

    predictions = model.predict(datamodule, trainer)
    np_predictions = predictions.detach().cpu().numpy()

    dataframe = datamodule.export_predict_csv_basic(
        np_predictions,
        out_dir=str(fold_dir),
        out_name=f"predictions_test_dataset_{fold}",
        return_csv=True,
    )

    output_path = fold_dir / f"predictions_test_dataset_{fold}.csv"
    dataframe.to_csv(output_path)

    return output_path

def run_crossval_inference(cfg: DictConfig) -> None:
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    all_targets = []
    all_predictions = []

    for fold in range(5):
        fold_dir = output_dir / f"fold_{fold}"
        checkpoint_path = fold_dir / "best.ckpt"

        datamodule = create_datamodule(cfg, fold)
        datamodule.setup(stage="test")

        base_system = RegressionSystem(
            cfg.model,
            **cfg.optim,
        )

        model = RegressionSystem.load_from_checkpoint(
            str(checkpoint_path),
            model=base_system.model,
            hparams_preprocess=False,
            weights_dir=str(fold_dir),
            loss=cfg.optim.loss,
            metrics=cfg.optim.metrics,
            strict=True,
        )

        model.eval()
        model.cuda()

        predictions = []
        targets = []

        with torch.inference_mode():
            for tile, label in datamodule.test_dataloader():
                tile = tile.cuda(non_blocking=True)
                prediction = model(tile)

                predictions.append(
                    prediction.detach().cpu()
                )
                targets.append(label.detach().cpu())

        all_predictions.append(
            torch.cat(predictions).numpy()
        )
        all_targets.append(
            torch.cat(targets).numpy()
        )

        del model
        del base_system
        del datamodule

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    targets = np.concatenate(all_targets).reshape(-1)
    predictions = np.concatenate(all_predictions).reshape(-1)

    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    results = pd.DataFrame(
        {
            "target": targets,
            "prediction": predictions,
        }
    )

    results.to_csv(
        output_dir / "all_folds_predictions_vs_targets.csv",
        index=False,
    )

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    pd.Series(metrics).to_json(
        output_dir / "crossval_metrics.json",
        indent=2,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.5)

    bounds = [
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    ]

    plt.plot(bounds, bounds, "k--", linewidth=2)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(
        "Target vs Prediction — all folds\n"
        f"RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}"
    )
    plt.savefig(
        output_dir / "all_folds_target_vs_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config",)
def main(cfg: DictConfig) -> None:
    mode = cfg.run.mode

    if mode == "train":
        score = train_fold(
            cfg=cfg,
            fold=cfg.run.fold,
        )
        print(f"Best validation MSE: {score:.6f}")

    elif mode == "predict":
        output_path = predict_fold(
            cfg=cfg,
            fold=cfg.run.fold,
        )
        print(f"Predictions saved to {output_path}")

    elif mode == "crossval_inference":
        run_crossval_inference(cfg)

    elif mode == "hpo":
        run_hpo(cfg)

    else:
        raise ValueError(
            f"Unknown run mode: {mode!r}. "
            "Expected train, predict, crossval_inference, "
            "hpo or plot_dataset."
        )


if __name__ == "__main__":
    main()
