"""NVIDIA DALI-backed DataModule for GPU-accelerated data loading.

The entire pipeline -- tar I/O, reshape, normalize, random flip -- runs
on GPU via DALI, eliminating Python per-sample overhead.  CPU threads
only handle sequential tar reads; everything else is a GPU kernel.

Standalone: no dependency on malpolon -- only needs Lightning, DALI, numpy.

Shard layout (created by ``create_shards.py``)::

    fold0-000000.tar, fold0-000001.tar, ...
    fold1-000000.tar, ...
    ...
    meta.json
    normalize.json

At training time, *fold* selects which groups are test / val / train:
    fold=k -> test=fold_k, val=fold_{k+1}, train=remaining folds
"""

import json
from pathlib import Path

import numpy as np
import lightning.pytorch as pl

from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import pandas as pd
from typing import Callable, Any, Union
from torch import Tensor


class _DALIWrapper:
    """Wraps DALIGenericIterator to yield (tile, label) tuples for Lightning."""

    def __init__(self, dali_iter):
        self._dali_iter = dali_iter

    def __iter__(self):
        for batch_list in self._dali_iter:
            d = batch_list[0]  # single GPU
            yield d["tile"], d["label"]

    def __len__(self):
        return len(self._dali_iter)


class DALIWebDatasetModule(pl.LightningDataModule):
    """Lightning DataModule using NVIDIA DALI for GPU-accelerated data loading.

    Reads from fold-based uint16 WebDataset shards (created by
    ``create_shards.py``).  The GPU pipeline is::

        reinterpret(uint16) -> reshape(28,H,W) -> cast(float32)
        -> normalize_with_folded_rescale -> random_flip

    The uint16-to-physical rescale (reflectance / 10000, thermal / 100)
    is folded into the normalization constants so there is no extra
    multiply on GPU.

    Parameters
    ----------
    wds_dir : str
        Directory containing ``fold*-*.tar``, ``meta.json``, and
        ``normalize.json`` (produced by ``create_shards.py``).
    fold : int
        Which fold to hold out as test (val = fold+1, train = rest).
    n_folds : int
        Total number of folds.  Must match the shards on disk.
    train_batch_size, inference_batch_size, num_workers, device_id
        Standard DataModule / DALI parameters.
    """

    def __init__(
        self,
        wds_dir: str,
        fold: int = 0,
        n_folds: int = 5,
        train_batch_size: int = 32,
        inference_batch_size: int = 16,
        num_workers: int = 4,
        device_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.wds_dir = Path(wds_dir).resolve()  # DALI needs absolute paths
        self.device_id = device_id

        # Load metadata
        self.meta = json.load(open(self.wds_dir / "meta.json"))
        self._shape = list(self.meta["shape"])  # [28, 224, 224]
        n_channels = self._shape[0]
        n_bands = self.meta["n_bands"]  # 7
        n_trimesters = self.meta["n_trimesters"]  # 4

        # Map split -> list of fold indices
        self._split_folds = {
            "test":  [fold % n_folds],
            "val":   [(fold + 1) % n_folds],
            "train": [(fold + i) % n_folds for i in range(2, n_folds)],
        }

        # Load 7-band normalize stats and fold rescale into them.
        # Shards store uint16: reflectance * 10000, thermal * 100.
        # Physical value = uint16 * scale.
        # Normalized = (physical - mean) / std = (uint16 - mean/scale) / (std/scale)
        with open(self.wds_dir / "normalize.json") as f:
            norm = json.load(f)
        mean_7 = np.array(norm["mean"], dtype=np.float64)
        std_7 = np.array(norm["std"], dtype=np.float64)

        n_reflectance = 6  # red, green, blue, nir08, swir16, swir22
        scale_7 = np.array(
            [1 / 10000.0] * n_reflectance + [1 / 100.0] * (n_bands - n_reflectance),
            dtype=np.float64,
        )

        # Fold rescale into normalization: work in uint16 space
        folded_mean_7 = mean_7 / scale_7
        folded_std_7 = std_7 / scale_7

        # Tile 7-band stats to 28 channels (4 trimesters x 7 bands)
        self._mean = np.tile(
            folded_mean_7.astype(np.float32), n_trimesters
        ).reshape(n_channels, 1, 1)
        self._std = np.tile(
            folded_std_7.astype(np.float32), n_trimesters
        ).reshape(n_channels, 1, 1)

        # Flat lists for fn.constant
        self._mean_list = self._mean.ravel().tolist()
        self._std_list = self._std.ravel().tolist()

    def _get_shard_paths(self, split):
        """Collect tar shard paths for all folds assigned to *split*.

        Returns (tar_paths, index_paths) where *index_paths* is a list of
        matching ``.idx`` paths if **every** shard has one, or ``None``
        otherwise.  The ``.idx`` files can be generated with::

            wds2idx <shard.tar> <shard.idx>
        """
        tar_paths = []
        for k in self._split_folds[split]:
            tar_paths.extend(sorted(self.wds_dir.glob(f"fold{k}-*.tar")))
        if not tar_paths:
            raise FileNotFoundError(
                f"No shards for split '{split}' (folds {self._split_folds[split]}) "
                f"in {self.wds_dir}"
            )
        tar_strs = [str(p) for p in tar_paths]

        # Auto-detect .idx index files (.tar replaced with .idx)
        idx_paths = [p.with_suffix(".idx") for p in tar_paths]
        if all(ip.exists() for ip in idx_paths):
            return tar_strs, [str(ip) for ip in idx_paths]
        return tar_strs, None

    def _build_pipeline(self, split, batch_size, is_train):
        shard_paths, index_paths = self._get_shard_paths(split)
        shape = self._shape
        n_channels = shape[0]
        mean_list = self._mean_list
        std_list = self._std_list
        num_threads = max(self.num_workers, 1)
        device_id = self.device_id

        @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
        def pipe():
            reader_kwargs = dict(
                paths=shard_paths,
                ext=["input", "target"],
                random_shuffle=is_train,
                name="reader",
            )
            if index_paths is not None:
                reader_kwargs["index_paths"] = index_paths
            raw_tile, raw_label = fn.readers.webdataset(**reader_kwargs)

            # GPU decode: raw bytes -> uint16 -> reshape -> float32
            tile = fn.reinterpret(raw_tile.gpu(), dtype=types.UINT16)
            tile = fn.reshape(tile, shape=shape)
            tile = fn.cast(tile, dtype=types.FLOAT)

            # Normalize (with uint16 rescale folded in)
            mean = fn.constant(fdata=mean_list, shape=[n_channels, 1, 1],
                               dtype=types.FLOAT, device="gpu")
            std = fn.constant(fdata=std_list, shape=[n_channels, 1, 1],
                              dtype=types.FLOAT, device="gpu")
            tile = (tile - mean) / std

            # Random flips for training
            if is_train:
                tile = fn.flip(
                    tile,
                    horizontal=fn.random.coin_flip(),
                    vertical=fn.random.coin_flip(),
                )

            # Label
            label = fn.reinterpret(raw_label.gpu(), dtype=types.FLOAT)
            label = fn.reshape(label, shape=[1])

            return tile, label

        p = pipe()
        p.build()
        return p

    def _make_loader(self, split, batch_size, is_train):
        pipe = self._build_pipeline(split, batch_size, is_train)
        return _DALIWrapper(
            DALIGenericIterator(
                pipe,
                ["tile", "label"],
                reader_name="reader",
                auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
        )

    # ------------------------------------------------------------------
    # Build once in setup(), return cached loaders thereafter.
    # With auto_reset=True the DALI iterators reset at epoch boundaries
    # without requiring a fresh pipeline build.
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if not hasattr(self, "_train_loader"):
                self._train_loader = self._make_loader(
                    "train", self.train_batch_size, is_train=True
                )
            if not hasattr(self, "_val_loader"):
                self._val_loader = self._make_loader(
                    "val", self.inference_batch_size, is_train=False
                )
        if stage in (None, "test"):
            if not hasattr(self, "_test_loader"):
                self._test_loader = self._make_loader(
                    "test", self.inference_batch_size, is_train=False
                )

    def train_dataloader(self):
        return self._make_loader("train", self.train_batch_size, is_train=True)

    def val_dataloader(self):
        return self._make_loader("val", self.inference_batch_size, is_train=False)

    def test_dataloader(self):
        return self._make_loader("test", self.inference_batch_size, is_train=False)

    def predict_dataloader(self):
        return self._make_loader("test", self.inference_batch_size, is_train=False)

    

