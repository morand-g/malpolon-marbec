import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from terratorch.models import EncoderDecoderFactory


# -----------------------------
# 1) Données synthétiques (régression scalaire)
#    y = sum_k w_k * mean(x_k) + c + bruit
# -----------------------------
class SyntheticRegressionDM(pl.LightningDataModule):
    def __init__(self, n_samples=2000, h=128, w=128, in_ch=6, batch_size=16):
        super().__init__()
        self.n_samples = n_samples
        self.h = h
        self.w = w
        self.in_ch = in_ch
        self.batch_size = batch_size

    def setup(self, stage=None):
        X = torch.zeros(self.n_samples, self.in_ch, self.h, self.w)
        y = torch.zeros(self.n_samples, 1)
        for i in range(self.n_samples // 2):
            X[i] = torch.ones(self.in_ch, self.h, self.w)
            y[i] = 1.0
        for i in range(self.n_samples // 2, self.n_samples):
            X[i] = torch.ones(self.in_ch, self.h, self.w) * -1.0
            y[i] = -1.0
        r = torch.randperm(self.n_samples)
        X = X[r]
        y = y[r]

        self.X_train, self.y_train = X[:4 * self.n_samples // 5], y[:4 * self.n_samples // 5]
        self.X_val, self.y_val = X[4 * self.n_samples // 5:], y[4 * self.n_samples // 5:]

    def train_dataloader(self):
        ds = TensorDataset(self.X_train, self.y_train)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        ds = TensorDataset(self.X_val, self.y_val)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)


# ------------------------------------
# 2) Modèle TerraTorch + wrapper régression
#     - sortie (B,1,H,W) -> avg pool -> (B,1)
# ------------------------------------
factory = EncoderDecoderFactory()
base_model = factory.build_model(
    task="classification",
    backbone="prithvi_eo_v1_100",
    backbone_freeze_backbone=True,
    backbone_pretrained=True,
    decoder="FCNDecoder",
    num_classes=1,
    backbone_in_channels=6,  # correspond aux 6 canaux des données
)


class TerraRegressor(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        y_map = self.model(x).output  # attendu: (B, 1, H, W) (segmentation-like)
        if y_map.ndim == 4:
            y = y_map.mean(dim=(2, 3))  # -> (B, 1)
        elif y_map.ndim == 2:
            y = y_map if y_map.size(1) == 1 else y_map.mean(dim=1, keepdim=True)
        else:
            y = y_map.view(y_map.size(0), -1).mean(dim=1, keepdim=True)
        return y

    @staticmethod
    def batch_r2(y_pred, y_true, eps=1e-8):
        # R2 sur le batch
        y_true_mean = y_true.mean()
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true_mean) ** 2).sum().clamp_min(eps)
        return 1.0 - ss_res / ss_tot

    def _log_scatter(self, tag, y_pred, y_true, global_step):
        # petit scatter prédiction vs cible (loggé 1x/epoch)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.scatter(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), s=8, alpha=0.6)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title(tag)
        self.logger.experiment.add_figure(tag, fig, global_step=global_step)
        plt.close(fig)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        r2 = self.batch_r2(y_pred, y_true)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/r2", r2, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y_true)
        r2 = self.batch_r2(y_pred, y_true)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/r2", r2, prog_bar=True, on_epoch=True)
        # log quelques histos pour le fun
        if batch_idx == 0:
            self.logger.experiment.add_histogram("val/y_true", y_true, global_step=self.global_step)
            self.logger.experiment.add_histogram("val/y_pred", y_pred, global_step=self.global_step)
        return {"val_loss": loss, "val_r2": r2, "y_pred": y_pred.detach(), "y_true": y_true.detach()}

    def on_validation_epoch_end(self):
        # scatter prédiction vs cible avec le 1er batch de val
        outs = self.trainer.callback_metrics
        # On ne récupère pas ici les batches, donc on refait un mini passage sur 1 batch de val pour tracer
        val_loader = self.trainer.datamodule.val_dataloader()
        x, y_true = next(iter(val_loader))
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        with torch.no_grad():
            y_pred = self.forward(x)
        self._log_scatter("val/scatter_y", y_pred, y_true, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -----------------------------
# 3) Entraînement + TensorBoard
# -----------------------------
if __name__ == "__main__":
    pl.seed_everything(7)

    dm = SyntheticRegressionDM(
        n_samples=500,  # augmente si tu veux une courbe plus lisse
        h=128, w=128,
        in_ch=6,
        batch_size=16,
    )
    dm.setup()

    model = TerraRegressor(base_model, lr=1e-3)

    logger = TensorBoardLogger(save_dir="tb_logs", name="terra_reg")

    trainer = pl.Trainer(
        max_epochs=10,  # 5-10 epochs suffisent pour voir la baisse de perte
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=5,
        enable_checkpointing=False
    )
    trainer.fit(model, dm)

    print("\nTensorBoard logs écrits dans:", logger.log_dir)
    print("Lance: tensorboard --logdir tb_logs\n")

    factory = EncoderDecoderFactory()
    base_model = factory.build_model(
        task="classification",
        backbone="prithvi_eo_v1_100",
        backbone_freeze_backbone=True,
        backbone_pretrained=True,
        decoder="FCNDecoder",
        num_classes=1,
        backbone_in_channels=6,  # correspond aux 6 canaux des données
    )




